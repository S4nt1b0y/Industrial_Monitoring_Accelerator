#!/usr/bin/env python3
"""Top-level LMS + FFT + MDC motor classifier reference model.

Pipeline:
    Q1.15 vibration dataset
      -> one delayed-reference LMS per vibration channel
      -> 64-point FFT per filtered channel
      -> bins 0..32 magnitude features
      -> top-3 peak bin indices per channel
      -> MDC features per channel
      -> decision-tree classifier
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from lms import LMSHardwareModel
from mdc import processar_tres_picos
from ml_classifier import (
    DEFAULT_DATASET,
    FIRST_FEATURE_BIN,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LAST_FEATURE_BIN,
    PROCESSED_DATASET_DIR,
    Q15_MAX,
    Q15_SCALE,
    VIBRATION_COLUMNS,
    export_tree_q15,
    make_fft_matrix,
    quantize_q15,
    require_power_of_two,
    resolve_dataset,
    split_data,
    valid_q15_datasets,
)


DEFAULT_OUTPUT_ROOT = Path("03.Reference/artifacts/top_classifier")
DEFAULT_FS_HZ = 6400
DEFAULT_MIN_K = 2
DEFAULT_LMS_DELAY = 1
LMS_NUM_TAPS = 8
LMS_MU = 0.01
FFT_FEATURE_TYPE = "lms_filtered_fft_approx_magnitude_abs_real_plus_abs_imag_q15"
MDC_FEATURE_TYPES = ("mdc_k0", "mdc_f0_hz", "mdc_result_valid")
FEATURE_BIN_COUNT = LAST_FEATURE_BIN - FIRST_FEATURE_BIN + 1
FFT_FEATURE_COUNT = len(VIBRATION_COLUMNS) * FEATURE_BIN_COUNT
MDC_FEATURE_COUNT = len(VIBRATION_COLUMNS) * len(MDC_FEATURE_TYPES)
TOTAL_FEATURE_COUNT = FFT_FEATURE_COUNT + MDC_FEATURE_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a top-level classifier from Q1.15 vibration windows after "
            "delayed-reference LMS filtering, FFT feature extraction and MDC "
            "features from the three largest FFT peaks."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Q1.15 Parquet dataset to train with. Accepts a file name from "
            "07.Datasets/processed or a path inside that directory."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List valid Q1.15 Parquet datasets from 07.Datasets/processed and exit.",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--lms-delay", type=int, default=DEFAULT_LMS_DELAY)
    parser.add_argument("--fs-hz", type=int, default=DEFAULT_FS_HZ)
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=16)
    parser.add_argument("--max-windows-per-class", type=int, default=20_000)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Artifact output directory. Defaults to "
            "03.Reference/artifacts/top_classifier/<dataset_stem>."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Parquet rows to read per streaming batch.",
    )
    return parser.parse_args()


def default_output_dir(dataset: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / dataset.stem


def list_datasets() -> None:
    datasets = valid_q15_datasets()
    if not datasets:
        print(f"No valid Q1.15 Parquet datasets found in {PROCESSED_DATASET_DIR}")
        return

    print(f"Valid Q1.15 datasets in {PROCESSED_DATASET_DIR}:")
    for dataset in datasets:
        parquet_file = pq.ParquetFile(dataset)
        print(f"- {dataset.name} ({parquet_file.metadata.num_rows} rows)")


def validate_args(args: argparse.Namespace) -> None:
    require_power_of_two(args.window_size)
    if args.lms_delay <= 0:
        raise ValueError(f"lms-delay must be greater than zero, got {args.lms_delay}")
    if args.lms_delay >= args.window_size:
        raise ValueError(
            f"lms-delay must be smaller than window-size, got "
            f"{args.lms_delay} >= {args.window_size}"
        )
    if args.fs_hz <= 0:
        raise ValueError(f"fs-hz must be greater than zero, got {args.fs_hz}")
    if args.min_k < 0:
        raise ValueError(f"min-k must be non-negative, got {args.min_k}")


def apply_lms_delayed_reference(
    signal_q15: np.ndarray,
    delay: int,
    num_taps: int = LMS_NUM_TAPS,
    mu: float = LMS_MU,
) -> np.ndarray:
    signal_float = signal_q15.astype(np.float64) / Q15_SCALE
    filtered = np.empty_like(signal_float)
    lms = LMSHardwareModel(num_taps=num_taps, mu=mu)

    for index, desired in enumerate(signal_float):
        reference = signal_float[index - delay] if index >= delay else 0.0
        result = lms.process_sample(x_new=float(reference), d_new=float(desired))
        filtered[index] = result["y"]

    return filtered


def three_largest_peak_bins(magnitude_bins: np.ndarray) -> np.ndarray:
    if magnitude_bins.shape[0] < 3:
        raise ValueError("At least three FFT bins are required for peak selection")
    peak_indices = np.argsort(-magnitude_bins, kind="stable")[:3]
    return np.sort(peak_indices.astype(np.int16))


def channel_fft_and_mdc_features(
    filtered_channel: np.ndarray,
    fft_matrix: np.ndarray,
    fs_hz: int,
    min_k: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fft_values = filtered_channel @ fft_matrix.T
    real_q15 = quantize_q15(fft_values.real)[FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(
        np.int32
    )
    imag_q15 = quantize_q15(fft_values.imag)[FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(
        np.int32
    )
    magnitude = np.minimum(np.abs(real_q15) + np.abs(imag_q15), Q15_MAX).astype(np.int16)

    peak_bins = three_largest_peak_bins(magnitude)
    k0, f0, result_valid = processar_tres_picos(
        int(peak_bins[0]),
        int(peak_bins[1]),
        int(peak_bins[2]),
        fs_hz=fs_hz,
        min_k=min_k,
        n_fft=n_fft,
    )
    mdc_features = np.asarray([k0, f0, int(result_valid)], dtype=np.int32)
    mdc_features = np.clip(mdc_features, 0, Q15_MAX).astype(np.int16)
    return magnitude, peak_bins, mdc_features


def window_to_top_features(
    window: np.ndarray,
    fft_matrix: np.ndarray,
    lms_delay: int,
    fs_hz: int,
    min_k: int,
) -> np.ndarray:
    fft_features = []
    mdc_features = []

    for channel_index in range(window.shape[1]):
        filtered_channel = apply_lms_delayed_reference(window[:, channel_index], lms_delay)
        channel_fft_features, _, channel_mdc_features = channel_fft_and_mdc_features(
            filtered_channel,
            fft_matrix,
            fs_hz=fs_hz,
            min_k=min_k,
            n_fft=window.shape[0],
        )
        fft_features.append(channel_fft_features)
        mdc_features.append(channel_mdc_features)

    return np.concatenate([*fft_features, *mdc_features]).astype(np.int16)


def windows_to_top_features(
    windows: np.ndarray,
    fft_matrix: np.ndarray,
    lms_delay: int,
    fs_hz: int,
    min_k: int,
) -> np.ndarray:
    features = np.empty((windows.shape[0], TOTAL_FEATURE_COUNT), dtype=np.int16)
    for window_index, window in enumerate(windows):
        features[window_index] = window_to_top_features(
            window,
            fft_matrix,
            lms_delay=lms_delay,
            fs_hz=fs_hz,
            min_k=min_k,
        )
    return features


def append_windows_from_run(
    run_values: np.ndarray,
    label: str,
    window_size: int,
    max_windows_per_class: int,
    fft_matrix: np.ndarray,
    lms_delay: int,
    fs_hz: int,
    min_k: int,
    features_by_class: dict[int, list[np.ndarray]],
) -> np.ndarray:
    class_id = LABEL_TO_ID[label]
    remaining = max_windows_per_class - len(features_by_class[class_id])
    if remaining <= 0:
        return run_values[:0]

    window_count = min(run_values.shape[0] // window_size, remaining)
    if window_count:
        stop = window_count * window_size
        windows = run_values[:stop].reshape(window_count, window_size, len(VIBRATION_COLUMNS))
        features_by_class[class_id].extend(
            windows_to_top_features(
                windows,
                fft_matrix,
                lms_delay=lms_delay,
                fs_hz=fs_hz,
                min_k=min_k,
            )
        )

    consumed = window_count * window_size
    return run_values[consumed:]


def collect_balanced_windows(
    dataset: Path,
    window_size: int,
    max_windows_per_class: int,
    batch_size: int,
    lms_delay: int,
    fs_hz: int,
    min_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["label"] + VIBRATION_COLUMNS
    parquet_file = pq.ParquetFile(dataset)
    fft_matrix = make_fft_matrix(window_size)
    features_by_class: dict[int, list[np.ndarray]] = {class_id: [] for class_id in ID_TO_LABEL}
    carry_label: str | None = None
    carry_values = np.empty((0, len(VIBRATION_COLUMNS)), dtype=np.int16)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        labels = frame["label"].to_numpy()
        values = frame[VIBRATION_COLUMNS].to_numpy(dtype=np.int16, copy=False)
        start = 0

        while start < len(labels):
            label = labels[start]
            stop = start + 1
            while stop < len(labels) and labels[stop] == label:
                stop += 1

            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown label in dataset: {label}")

            run_values = values[start:stop]
            if carry_label == label and carry_values.size:
                run_values = np.vstack([carry_values, run_values])
            elif carry_label != label:
                carry_values = carry_values[:0]

            carry_label = label
            carry_values = append_windows_from_run(
                run_values,
                label,
                window_size,
                max_windows_per_class,
                fft_matrix,
                lms_delay,
                fs_hz,
                min_k,
                features_by_class,
            )
            start = stop

        counts = {class_id: len(items) for class_id, items in features_by_class.items()}
        if all(count >= max_windows_per_class for count in counts.values()):
            break

    missing = {
        ID_TO_LABEL[class_id]: max_windows_per_class - len(items)
        for class_id, items in features_by_class.items()
        if len(items) < max_windows_per_class
    }
    if missing:
        raise ValueError(f"Not enough complete windows for the requested balance: {missing}")

    features = []
    labels = []
    for class_id, class_features in features_by_class.items():
        features.extend(class_features)
        labels.extend([class_id] * len(class_features))

    feature_array = np.asarray(features, dtype=np.int16)
    if feature_array.shape[1] != TOTAL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {TOTAL_FEATURE_COUNT} features, got {feature_array.shape[1]}"
        )
    return feature_array, np.asarray(labels, dtype=np.int8)


def make_feature_map(window_size: int) -> list[dict[str, Any]]:
    rows = []
    feature_index = 0

    for channel in VIBRATION_COLUMNS:
        for bin_index in range(FIRST_FEATURE_BIN, LAST_FEATURE_BIN + 1):
            rows.append(
                {
                    "feature_index": feature_index,
                    "channel": channel,
                    "bin_fft": bin_index,
                    "feature_type": FFT_FEATURE_TYPE,
                }
            )
            feature_index += 1

    for channel in VIBRATION_COLUMNS:
        for feature_type in MDC_FEATURE_TYPES:
            rows.append(
                {
                    "feature_index": feature_index,
                    "channel": channel,
                    "bin_fft": "",
                    "feature_type": feature_type,
                }
            )
            feature_index += 1

    if feature_index != TOTAL_FEATURE_COUNT:
        raise ValueError(f"Expected {TOTAL_FEATURE_COUNT} mapped features, got {feature_index}")
    return rows


def export_feature_map(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def class_counts(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(class_id): int(counts.get(class_id, 0)) for class_id in ID_TO_LABEL}


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    dataset = resolve_dataset(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = collect_balanced_windows(
        dataset,
        args.window_size,
        args.max_windows_per_class,
        args.batch_size,
        args.lms_delay,
        args.fs_hz,
        args.min_k,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        features,
        labels,
        args.test_size,
        args.val_size,
        args.seed,
    )

    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = {
        "dataset": {
            "name": dataset.name,
            "path": str(dataset),
            "q_format": "q1_15",
        },
        "artifact_dir": str(output_dir),
        "class_map": {str(class_id): label for class_id, label in ID_TO_LABEL.items()},
        "input_columns": VIBRATION_COLUMNS,
        "pipeline": [
            "delayed_reference_lms_per_channel",
            "fft_dif_radix2_per_filtered_channel",
            "bins_0_to_32_magnitude",
            "top_3_peak_bins_per_channel",
            "mdc_per_channel",
            "decision_tree_classifier",
        ],
        "window_size": args.window_size,
        "lms": {
            "num_taps": LMS_NUM_TAPS,
            "mu": LMS_MU,
            "reference": "same_channel_delayed_by_lms_delay_samples",
            "desired": "current_same_channel_sample",
            "output_used": "y",
            "delay_samples": args.lms_delay,
        },
        "selected_fft_bins": {
            "first": FIRST_FEATURE_BIN,
            "last": LAST_FEATURE_BIN,
            "count_per_channel": FEATURE_BIN_COUNT,
        },
        "mdc": {
            "peak_count_per_channel": 3,
            "peak_source": "indices_of_three_largest_fft_magnitude_bins",
            "fs_hz": args.fs_hz,
            "min_k": args.min_k,
            "features_per_channel": list(MDC_FEATURE_TYPES),
        },
        "feature_count": int(features.shape[1]),
        "feature_layout": {
            "fft_features": FFT_FEATURE_COUNT,
            "mdc_features": MDC_FEATURE_COUNT,
            "total": TOTAL_FEATURE_COUNT,
        },
        "fft_q_format": "q1_15",
        "fft_scale": "fft_output_divided_by_window_size_before_q15_quantization",
        "classifier_output": "class_id_integer_0_to_3",
        "sample_counts": {
            "total": class_counts(labels),
            "train": class_counts(y_train),
            "validation": class_counts(y_val),
            "test": class_counts(y_test),
        },
        "model": {
            "type": "DecisionTreeClassifier",
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "node_count": int(model.tree_.node_count),
            "depth": int(model.tree_.max_depth),
        },
        "cross_validation": {
            "folds": args.cv_folds,
            "accuracy_scores": cv_scores.tolist(),
            "accuracy_mean": float(np.mean(cv_scores)),
            "accuracy_std": float(np.std(cv_scores)),
        },
        "validation": {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "confusion_matrix": confusion_matrix(y_val, val_pred, labels=list(ID_TO_LABEL)).tolist(),
            "classification_report": classification_report(
                y_val,
                val_pred,
                labels=list(ID_TO_LABEL),
                target_names=[ID_TO_LABEL[index] for index in ID_TO_LABEL],
                output_dict=True,
                zero_division=0,
            ),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "confusion_matrix": confusion_matrix(y_test, test_pred, labels=list(ID_TO_LABEL)).tolist(),
            "classification_report": classification_report(
                y_test,
                test_pred,
                labels=list(ID_TO_LABEL),
                target_names=[ID_TO_LABEL[index] for index in ID_TO_LABEL],
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    export_tree_q15(model, output_dir / "tree_q15.json")
    export_feature_map(output_dir / "feature_map.csv", make_feature_map(args.window_size))

    print(f"Collected balanced windows: {class_counts(labels)}")
    print(f"Dataset: {dataset.name}")
    print(f"Feature count: {features.shape[1]}")
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Wrote artifacts to {output_dir}")
    return metrics


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        list_datasets()
        return
    train(args)


if __name__ == "__main__":
    main()
