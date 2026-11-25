#!/usr/bin/env python3
"""Infer a masked LSTM, BiLSTM, MLP, TCN or Transformer on prepared NPZ data.

The script reproduces the feature/target normalization and causal windowing
stored in a training checkpoint.  It predicts every window that stays inside
one continuous segment.  References and metrics are restricted to
``target_mask``; unknown ground contacts are therefore never scored as zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if "--show" in sys.argv:
    # This workstation may configure Agg as its global default even when a
    # graphical DISPLAY is available. Tkinter is the lightest interactive
    # backend used here and is explicitly selected before importing pyplot.
    matplotlib.use("TkAgg", force=True)
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset

from grfm_schema import participant_of_subject
from train_lstm_two_feet_masked import (
    SensorDropout,
    append_sensor_presence_names,
    build_model,
    gauge_indexes,
    inactive_gauge_columns,
    loadsol_inputs,
)


UNITS = {"Fx": "N", "Fy": "N", "Fz": "N", "Mx": "N·m", "My": "N·m", "Mz": "N·m"}


@dataclass
class PreparedAcquisition:
    name: str
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    time_s: np.ndarray
    segment_id: np.ndarray
    # ``--sensor-dropout`` checkpoints: the full configuration, Inactive sensors removed.
    sensor_dropout: SensorDropout | None = None
    inactive_columns: tuple[int, ...] = ()


class InferenceWindowDataset(Dataset):
    """All causal/centred windows contained in one uninterrupted segment."""

    def __init__(
        self,
        acquisition: PreparedAcquisition,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        window: int,
        causal: bool,
    ) -> None:
        self.acquisition = acquisition
        self.x_mean = x_mean
        self.x_std = x_std
        self.window = window
        self.causal = causal
        left = window - 1 if causal else window // 2
        right = window - left - 1
        self.target_indexes: list[int] = []
        for target_index in range(left, len(acquisition.x) - right):
            start = target_index - left
            stop = target_index + right + 1
            if (
                acquisition.segment_id[start]
                == acquisition.segment_id[target_index]
                == acquisition.segment_id[stop - 1]
            ):
                self.target_indexes.append(target_index)

    def __len__(self) -> int:
        return len(self.target_indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        target_index = self.target_indexes[item]
        start = (
            target_index - self.window + 1
            if self.causal
            else target_index - self.window // 2
        )
        x = self.acquisition.x[start : start + self.window]
        x = (x - self.x_mean) / self.x_std
        if self.acquisition.sensor_dropout is not None:
            x = self.acquisition.sensor_dropout.apply(x, self.acquisition.inactive_columns)
        return torch.from_numpy(x), target_index


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demandé mais indisponible")
    return torch.device(requested)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"checkpoint illisible {path}: {exc}") from exc
    required = {
        "state_dict",
        "input_size",
        "hidden_size",
        "layers",
        "dropout",
        "window_samples",
        "causal",
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "feature_names",
        "targets",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"champs absents du checkpoint: {missing}")
    if bool(checkpoint.get("hard_airborne_zero", False)):
        print(
            "ATTENTION: ce checkpoint a ete entraine avec hard_airborne_zero; "
            "le label mocap hors ligne n'est pas applique par ce script.",
            file=sys.stderr,
        )
    return checkpoint


def load_metadata(path: Path) -> dict:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata illisible {path}: {exc}") from exc
    if not isinstance(metadata.get("feature_names"), list):
        raise ValueError(f"feature_names absent de {path}")
    if not isinstance(metadata.get("targets"), list):
        raise ValueError(f"targets absent de {path}")
    return metadata


def enrich_gauge_inputs(
    x: np.ndarray,
    feature_names: list[str],
    epsilon_n: float,
) -> tuple[np.ndarray, list[str]]:
    """Rebuild the gauge sums/proportions exactly as during training."""
    left = gauge_indexes(feature_names, "L")
    right = gauge_indexes(feature_names, "R")
    if not left or not right:
        raise ValueError("colonnes de jauges gauche/droite introuvables")
    additions: list[np.ndarray] = []
    enriched_names = list(feature_names)
    for side, indexes in (("L", left), ("R", right)):
        values = x[:, indexes]
        total = values.sum(axis=1, keepdims=True, dtype=np.float32)
        safe = np.where(np.abs(total) >= epsilon_n, total, np.nan)
        proportions = np.divide(
            values,
            safe,
            out=np.zeros_like(values, dtype=np.float32),
            where=np.isfinite(safe),
        )
        additions.extend((total.astype(np.float32), proportions))
        enriched_names.append(f"{side}_gauge_sum_N")
        enriched_names.extend(f"{feature_names[index]}_fraction" for index in indexes)
    return np.concatenate((x, *additions), axis=1), enriched_names


def estimate_load_from_inputs(
    x: np.ndarray,
    time_s: np.ndarray,
    feature_names: list[str],
    calibration_seconds: float,
) -> float:
    """Reproduce the deployment-compatible load estimate stored by training."""
    force_indexes = gauge_indexes(feature_names)
    if not force_indexes:
        raise ValueError("colonnes de jauges introuvables pour estimer la charge")
    elapsed = time_s - time_s[0]
    selected = elapsed <= calibration_seconds
    if not selected.any():
        selected = np.ones(len(time_s), dtype=bool)
    total = x[selected][:, force_indexes].sum(axis=1, dtype=np.float64)
    finite = total[np.isfinite(total)]
    if not len(finite):
        raise ValueError("charge semelle non estimable")
    estimate = float(np.median(finite))
    if estimate <= 1.0:
        raise ValueError(f"charge semelle non physique: {estimate:.3f} N")
    return estimate


def load_acquisition(
    path: Path, checkpoint: dict, metadata: dict
) -> tuple[PreparedAcquisition, float]:
    expected_features = list(checkpoint["feature_names"])
    source_features = list(checkpoint.get("source_feature_names", expected_features))
    actual_features = list(metadata["feature_names"])
    missing_features = [name for name in source_features if name not in actual_features]
    if missing_features:
        raise ValueError(f"{path.name}: features manquantes: {missing_features}")
    feature_indexes = [actual_features.index(name) for name in source_features]

    expected_targets = list(checkpoint["targets"])
    actual_targets = list(metadata["targets"])
    missing_targets = [name for name in expected_targets if name not in actual_targets]
    if missing_targets:
        raise ValueError(f"{path.name}: cibles manquantes: {missing_targets}")
    target_indexes = [actual_targets.index(name) for name in expected_targets]

    try:
        with np.load(path) as values:
            required = {"X", "y", "target_mask", "target_time_s", "segment_id"}
            missing = sorted(required - set(values.files))
            if missing:
                raise ValueError(f"{path.name}: tableaux absents: {missing}")
            source_x = values["X"][:, feature_indexes].astype(np.float32)
            y = values["y"][:, target_indexes].astype(np.float32)
            mask = values["target_mask"][:, target_indexes].astype(bool)
            time_s = values["target_time_s"].astype(np.float64)
            segment_id = values["segment_id"].astype(np.int64)
    except OSError as exc:
        raise ValueError(f"NPZ illisible {path}: {exc}") from exc

    mode = str(checkpoint.get("input_features", "raw"))
    if mode == "raw":
        x, rebuilt_names = source_x, source_features
    elif mode == "gauge_augmented":
        x, rebuilt_names = enrich_gauge_inputs(
            source_x,
            source_features,
            float(checkpoint.get("gauge_proportion_epsilon_n", 1.0)),
        )
    elif mode == "loadsol":
        x, rebuilt_names = loadsol_inputs(source_x, source_features, bool(checkpoint.get("loadsol_heel_imu", False)))
    else:
        raise ValueError(f"mode de features inconnu dans le checkpoint: {mode}")
    sensor_dropout, inactive_columns = None, ()
    if checkpoint.get("sensor_dropout"):
        # Same as the engine: presence columns appended (0 in the data, written per window).
        inactive_columns = inactive_gauge_columns(x, rebuilt_names)
        x = np.concatenate((x, np.zeros((len(x), len(checkpoint["sensor_dropout"]["groups"])), np.float32)), axis=1)
        rebuilt_names = append_sensor_presence_names(rebuilt_names)
        sensor_dropout = SensorDropout.from_feature_names(rebuilt_names)
    if rebuilt_names != expected_features:
        raise ValueError(
            f"{path.name}: ordre des features reconstruites incompatible avec le checkpoint"
        )

    n = len(x)
    if x.shape != (n, int(checkpoint["input_size"])):
        raise ValueError(f"{path.name}: X a la forme incompatible {x.shape}")
    if y.shape != (n, len(expected_targets)) or mask.shape != y.shape:
        raise ValueError(f"{path.name}: formes y/mask incompatibles")
    if time_s.shape != (n,) or segment_id.shape != (n,):
        raise ValueError(f"{path.name}: formes temps/segment incompatibles")
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
        raise ValueError(f"{path.name}: NaN/Inf dans une valeur utilisee")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"{path.name}: temps non strictement croissant")
    target_scaling = str(checkpoint.get("target_scaling", "standard"))
    if target_scaling == "standard":
        load_estimate_n = 1.0
    elif target_scaling == "subject_load":
        load_estimate_n = estimate_load_from_inputs(
            source_x,
            time_s,
            source_features,
            float(checkpoint.get("load_calibration_seconds", 3.0)),
        )
    else:
        raise ValueError(
            f"normalisation des cibles inconnue dans le checkpoint: {target_scaling}"
        )
    return (
        PreparedAcquisition(path.stem, x, y, mask, time_s, segment_id, sensor_dropout, inactive_columns),
        load_estimate_n,
    )


def infer(
    model: torch.nn.Module,
    dataset: InferenceWindowDataset,
    loader: DataLoader,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    indexes: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, target_index in loader:
            normalized = model(x.to(device)).cpu().numpy()
            predictions.append(normalized * y_std + y_mean)
            indexes.append(target_index.numpy())
    return np.concatenate(indexes), np.vstack(predictions)


def metric(reference: np.ndarray, prediction: np.ndarray) -> tuple[int, float, float]:
    finite = np.isfinite(reference) & np.isfinite(prediction)
    reference = reference[finite]
    prediction = prediction[finite]
    n = len(reference)
    if not n:
        return 0, math.nan, math.nan
    rmse = float(np.sqrt(np.mean(np.square(reference - prediction))))
    cc = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if n > 1 and np.std(reference) > 0 and np.std(prediction) > 0
        else math.nan
    )
    return n, rmse, cc


def zero_phase_filter_prediction(
    prediction: np.ndarray,
    time_s: np.ndarray,
    segment_id: np.ndarray,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Offline low-pass filtering, independently inside each data segment."""
    filtered = prediction.copy()
    for segment in np.unique(segment_id):
        selected = np.flatnonzero(segment_id == segment)
        if len(selected) < 2:
            print(
                f"ATTENTION: segment {segment} trop court pour etre filtre; "
                "prediction brute conservee.",
                file=sys.stderr,
            )
            continue
        rate_hz = float(1.0 / np.median(np.diff(time_s[selected])))
        if cutoff_hz >= rate_hz / 2.0:
            raise ValueError(
                f"frequence de coupure {cutoff_hz:g} Hz >= Nyquist "
                f"{rate_hz / 2.0:.3f} Hz dans le segment {segment}"
            )
        sos = butter(order, cutoff_hz, btype="lowpass", fs=rate_hz, output="sos")
        try:
            filtered[selected] = sosfiltfilt(sos, prediction[selected], axis=0)
        except ValueError:
            print(
                f"ATTENTION: segment {segment} ({len(selected)} points) trop court "
                "pour filtfilt; prediction brute conservee.",
                file=sys.stderr,
            )
    return filtered


def zero_phase_filter_reference(
    reference: np.ndarray,
    mask: np.ndarray,
    time_s: np.ndarray,
    segment_id: np.ndarray,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Filter each continuous valid-reference interval independently.

    Unknown labels must not be interpreted as zeros by the filter. Each target
    is therefore filtered only within contiguous runs where target_mask is
    true and which remain inside one acquisition segment.
    """
    filtered = reference.copy()
    sample_indexes = np.arange(len(reference))
    for target_index in range(reference.shape[1]):
        valid = mask[:, target_index]
        if not valid.any():
            continue
        boundaries = (
            np.r_[True, ~valid[:-1]]
            | np.r_[True, segment_id[1:] != segment_id[:-1]]
            | np.r_[True, np.diff(sample_indexes) != 1]
        )
        starts = np.flatnonzero(valid & boundaries)
        for start in starts:
            stop = start + 1
            while (
                stop < len(reference)
                and valid[stop]
                and segment_id[stop] == segment_id[start]
            ):
                stop += 1
            selected = np.arange(start, stop)
            if len(selected) < 2:
                continue
            rate_hz = float(1.0 / np.median(np.diff(time_s[selected])))
            if cutoff_hz >= rate_hz / 2.0:
                raise ValueError(
                    f"frequence de coupure {cutoff_hz:g} Hz >= Nyquist "
                    f"{rate_hz / 2.0:.3f} Hz dans une reference valide"
                )
            sos = butter(order, cutoff_hz, btype="lowpass", fs=rate_hz, output="sos")
            try:
                filtered[selected, target_index] = sosfiltfilt(
                    sos, reference[selected, target_index]
                )
            except ValueError:
                # Very short contact/airborne labels cannot support filtfilt's
                # padding. Keeping them unchanged is safer than mixing them
                # with neighbouring unknown labels.
                pass
    return filtered


def save_foot_plot(
    output: Path,
    acquisition: PreparedAcquisition,
    indexes: np.ndarray,
    prediction: np.ndarray,
    reference_values: np.ndarray,
    targets: list[str],
    side: str,
    model_name: str,
    variant: str,
    show: bool,
) -> list[dict[str, object]]:
    target_indexes = [i for i, name in enumerate(targets) if name.endswith(f"_{side}")]
    if len(target_indexes) != 6:
        raise ValueError(f"6 sorties attendues pour le pied {side}, trouve {len(target_indexes)}")

    time_s = acquisition.time_s[indexes]
    segments = acquisition.segment_id[indexes]
    figure, axes = plt.subplots(3, 2, figsize=(17, 12), sharex=True, constrained_layout=True)
    rows: list[dict[str, object]] = []
    for plot_index, target_index in enumerate(target_indexes):
        axis = axes.ravel()[plot_index]
        target = targets[target_index]
        valid = acquisition.mask[indexes, target_index]
        reference = reference_values[:, target_index]
        estimate = prediction[:, target_index]
        n, rmse, cc = metric(reference[valid], estimate[valid])
        rows.append(
            {
                "trial": acquisition.name,
                "variant": variant,
                "foot": side,
                "target": target,
                "n_valid": n,
                "rmse": rmse,
                "cc": cc,
                "unit": UNITS[target[:2]],
            }
        )

        # Plot each uninterrupted segment separately to avoid drawing a line
        # across packet-loss gaps. Prediction remains visible where the label
        # is unknown; the reference is drawn only where target_mask is true.
        for segment_number, segment in enumerate(np.unique(segments)):
            selected = segments == segment
            axis.plot(
                time_s[selected],
                estimate[selected],
                color="tab:orange",
                linewidth=0.9,
                alpha=0.9,
                label="Prediction" if segment_number == 0 else None,
            )
        for segment_number, segment in enumerate(np.unique(segments)):
            selected = segments == segment
            visible_reference = np.where(valid[selected], reference[selected], np.nan)
            axis.plot(
                time_s[selected],
                visible_reference,
                color="tab:blue",
                linewidth=1.0,
                label="Reference valide" if segment_number == 0 else None,
            )
        cc_text = f"{cc:.3f}" if np.isfinite(cc) else "n/a"
        rmse_text = f"{rmse:.2f}" if np.isfinite(rmse) else "n/a"
        axis.set_title(
            f"{target}   RMSE={rmse_text} {UNITS[target[:2]]}   CC={cc_text}   n={n}"
        )
        axis.set_ylabel(UNITS[target[:2]])
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Temps (s)")
    foot_name = "gauche" if side == "L" else "droit"
    figure.suptitle(
        f"{acquisition.name} — pied {foot_name} — {model_name} — {variant}",
        fontsize=15,
    )
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return rows


def identity_from_npz(path: Path) -> str:
    return participant_of_subject(path.stem.split("_", 1)[0])


def expand_inputs(inputs: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in inputs:
        if path.is_dir():
            result.extend(sorted(path.glob("*.npz")))
        elif path.is_file():
            result.append(path)
        else:
            raise ValueError(f"entree introuvable: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in result))
    if not unique:
        raise ValueError("aucun fichier NPZ trouve")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", type=Path, help="checkpoint .pt LSTM, BiLSTM, MLP, TCN ou Transformer"
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="fichier(s) NPZ ou dossier NPZ")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_npz"))
    parser.add_argument(
        "--data-metadata",
        type=Path,
        help="metadata.json; par defaut celui situe a cote de chaque NPZ",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--post-filter-hz",
        type=float,
        default=10.0,
        help="coupure du Butterworth zero-phase applique aux predictions (defaut: 10 Hz)",
    )
    parser.add_argument(
        "--post-filter-order",
        type=int,
        default=4,
        help="ordre du Butterworth de post-filtrage (defaut: 4)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="sauvegarde aussi temps, predictions, references et masques en NPZ compresse",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size doit etre positif et num-workers >= 0")
    if args.post_filter_hz <= 0 or args.post_filter_order < 1:
        parser.error("post-filter-hz et post-filter-order doivent etre positifs")
    if not args.model.is_file():
        parser.error(f"checkpoint introuvable: {args.model}")

    try:
        input_paths = expand_inputs(args.inputs)
        device = select_device(args.device)
        checkpoint = load_checkpoint(args.model, device)
        targets = list(checkpoint["targets"])
        model_type = str(checkpoint.get("model_type", "lstm"))
        model = build_model(
            model_type=model_type,
            input_size=int(checkpoint["input_size"]),
            window_samples=int(checkpoint["window_samples"]),
            hidden_size=int(checkpoint["hidden_size"]),
            layers=int(checkpoint["layers"]),
            dropout=float(checkpoint["dropout"]),
            output_size=len(targets),
            tcn_kernel_size=int(checkpoint.get("tcn_kernel_size", 3)),
            transformer_heads=int(checkpoint.get("transformer_heads", 4)),
            transformer_feedforward_size=int(
                checkpoint.get("transformer_feedforward_size", 128)
            ),
            feature_names=list(checkpoint.get("feature_names") or []),
            # The Foot height head is not used here, but its weights are in the checkpoint.
            foot_height_output=bool(checkpoint.get("foot_height_output", False)),
            patch_samples=int(checkpoint.get("patch_samples", 5)),
            decoder_layers=int(checkpoint.get("decoder_layers", 1)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    except ValueError as exc:
        parser.error(str(exc))

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)
    window = int(checkpoint["window_samples"])
    causal = bool(checkpoint["causal"])
    if x_mean.shape != (int(checkpoint["input_size"]),) or x_std.shape != x_mean.shape:
        parser.error("statistiques X incompatibles dans le checkpoint")
    if y_mean.shape != (len(targets),) or y_std.shape != y_mean.shape:
        parser.error("statistiques y incompatibles dans le checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    metadata_cache: dict[Path, dict] = {}
    test_group = checkpoint.get("test_group")
    train_groups = set(checkpoint.get("train_groups", []))

    for input_path in input_paths:
        subject = identity_from_npz(input_path)
        if test_group and subject != test_group:
            message = (
                f"ATTENTION {input_path.name}: checkpoint de test={test_group}, "
                f"mais le NPZ appartient a {subject}."
            )
            if subject in train_groups:
                message += " Ce sujet faisait partie du train: metriques non independantes."
            print(message, file=sys.stderr)

        metadata_path = args.data_metadata or input_path.parent / "metadata.json"
        metadata_path = metadata_path.resolve()
        try:
            metadata = metadata_cache.setdefault(metadata_path, load_metadata(metadata_path))
            acquisition, load_estimate_n = load_acquisition(
                input_path, checkpoint, metadata
            )
            dataset = InferenceWindowDataset(acquisition, x_mean, x_std, window, causal)
        except ValueError as exc:
            parser.error(str(exc))
        if not len(dataset):
            print(f"IGNORE {input_path.name}: aucune fenetre complete", file=sys.stderr)
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        indexes, prediction = infer(model, dataset, loader, y_mean, y_std, device)
        prediction *= load_estimate_n
        reference_raw = acquisition.y[indexes]
        reference_mask = acquisition.mask[indexes]
        try:
            filtered_prediction = zero_phase_filter_prediction(
                prediction,
                acquisition.time_s[indexes],
                acquisition.segment_id[indexes],
                args.post_filter_hz,
                args.post_filter_order,
            )
            filtered_reference = zero_phase_filter_reference(
                reference_raw,
                reference_mask,
                acquisition.time_s[indexes],
                acquisition.segment_id[indexes],
                args.post_filter_hz,
                args.post_filter_order,
            )
        except ValueError as exc:
            parser.error(f"{input_path.name}: {exc}")
        variants = (
            ("prediction_et_reference_brutes", "raw", prediction, reference_raw),
            (
                f"prediction_et_reference_filtrees_zero_phase_{args.post_filter_hz:g}Hz",
                f"filtered_{args.post_filter_hz:g}Hz",
                filtered_prediction,
                filtered_reference,
            ),
        )
        # Raw left/right figures are produced first, followed by the filtered
        # left/right figures, so --show presents the two processing variants
        # in an unambiguous order.
        for variant, file_variant, variant_prediction, variant_reference in variants:
            for side, suffix in (("L", "left"), ("R", "right")):
                plot_path = args.output_dir / (
                    f"{input_path.stem}_{args.model.stem}_{file_variant}_{suffix}.png"
                )
                all_metrics.extend(
                    save_foot_plot(
                        plot_path,
                        acquisition,
                        indexes,
                        variant_prediction,
                        variant_reference,
                        targets,
                        side,
                        args.model.stem,
                        variant,
                        args.show,
                    )
                )
                print(f"Figure: {plot_path}")
        if args.save_predictions:
            output = args.output_dir / f"{input_path.stem}_{args.model.stem}_predictions.npz"
            np.savez_compressed(
                output,
                target_time_s=acquisition.time_s[indexes],
                segment_id=acquisition.segment_id[indexes],
                prediction_raw=prediction,
                prediction_filtered=filtered_prediction,
                reference_raw=reference_raw,
                reference_filtered=filtered_reference,
                target_mask=acquisition.mask[indexes],
                targets=np.asarray(targets),
                post_filter_hz=np.asarray(args.post_filter_hz),
                post_filter_order=np.asarray(args.post_filter_order),
            )
            print(f"Predictions: {output}")

    if not all_metrics:
        parser.error("aucune acquisition exploitable")
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    run_info = {
        "model": str(args.model.resolve()),
        "inputs": [str(path) for path in input_paths],
        "device": str(device),
        "window_samples": window,
        "causal": causal,
        "post_filter": {
            "type": "Butterworth sosfiltfilt (offline, non-causal)",
            "cutoff_hz": args.post_filter_hz,
            "order": args.post_filter_order,
            "applied_per_segment": True,
            "filtered_signals": ["prediction", "reference"],
            "reference_filtering_respects_contiguous_target_mask": True,
        },
        "reference_metrics_use_target_mask": True,
        "prediction_is_computed_outside_reference_mask": True,
    }
    (args.output_dir / "inference_metadata.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Metriques: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
