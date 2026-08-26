#!/usr/bin/env python3
"""Train an RTL-friendly motor-state classifier from Q1.15 FFT features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from fft import bit_reverse_order, fft_dif_radix2


DEFAULT_DATASET = Path("07.Datasets/processed/motor_measurements_q15.parquet")
PROCESSED_DATASET_DIR = Path("07.Datasets/processed")
DEFAULT_OUTPUT_ROOT = Path("03.Reference/artifacts/ml_classifier")
VIBRATION_COLUMNS = [
    "aceleracao_x_mancal_a",
    "aceleracao_y_mancal_a",
    "aceleracao_x_mancal_b",
    "aceleracao_y_mancal_b",
]
LABEL_TO_ID = {
    "operacao_normal": 0,
    "desalinhamento": 1,
    "desbalanceamento": 2,
    "desgaste_rolamento": 3,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
Q15_SCALE = 32768.0
Q15_MIN = -32768
Q15_MAX = 32767
FIRST_FEATURE_BIN = 0
LAST_FEATURE_BIN = 32
FEATURE_BIN_COUNT = LAST_FEATURE_BIN - FIRST_FEATURE_BIN + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small decision-tree classifier from 64-sample Q1.15 FFT "
            "windows of the motor vibration dataset."
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
    parser.add_argument(
        "--compare-q15-datasets",
        action="store_true",
        help="Train one classifier for each valid Q1.15 dataset and report the best result.",
    )
    parser.add_argument("--window-size", type=int, default=64)
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
            "03.Reference/artifacts/ml_classifier/<dataset_stem>."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Parquet rows to read per streaming batch.",
    )
    return parser.parse_args()


def q15_dataset_name(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".parquet") and ("q15" in name or "q115" in name) and "q17" not in name


def is_q15_parquet(path: Path) -> bool:
    if not path.is_file() or not q15_dataset_name(path):
        return False

    try:
        schema = pq.read_schema(path)
    except Exception:
        return False

    required_columns = {"label", *VIBRATION_COLUMNS}
    if not required_columns.issubset(set(schema.names)):
        return False

    return all(pa.types.is_int16(schema.field(column).type) for column in VIBRATION_COLUMNS)


def valid_q15_datasets() -> list[Path]:
    return [
        path
        for path in sorted(PROCESSED_DATASET_DIR.glob("*.parquet"))
        if is_q15_parquet(path)
    ]


def resolve_dataset(path: Path) -> Path:
    candidate = path
    if not candidate.is_absolute() and candidate.parent == Path("."):
        candidate = PROCESSED_DATASET_DIR / candidate
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".parquet")

    candidate = candidate.resolve()
    processed_dir = PROCESSED_DATASET_DIR.resolve()
    if processed_dir not in candidate.parents:
        raise ValueError(f"Dataset must be inside {PROCESSED_DATASET_DIR}: {path}")
    if not is_q15_parquet(candidate):
        valid_names = ", ".join(dataset.name for dataset in valid_q15_datasets()) or "none"
        raise ValueError(f"Invalid Q1.15 dataset: {path}. Valid options: {valid_names}")
    return candidate


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


def require_power_of_two(value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"window-size must be a positive power of two, got {value}")
    if value < FEATURE_BIN_COUNT:
        raise ValueError(
            f"window-size must be at least {FEATURE_BIN_COUNT} "
            f"to use FFT bins {FIRST_FEATURE_BIN}..{LAST_FEATURE_BIN}, got {value}"
        )


def quantize_q15(values: np.ndarray) -> np.ndarray:
    quantized = np.rint(values * Q15_SCALE)
    return np.clip(quantized, Q15_MIN, Q15_MAX).astype(np.int16)


def fft_q15(window_q15: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window_float = window_q15.astype(np.float64) / Q15_SCALE
    scrambled = fft_dif_radix2(window_float)
    indices = bit_reverse_order(window_q15.shape[0])
    ordered = np.empty_like(scrambled)
    ordered[indices] = scrambled

    # A 1/N scale keeps the FFT reference in Q1.15 and avoids broad saturation.
    ordered = ordered / window_q15.shape[0]
    return quantize_q15(ordered.real), quantize_q15(ordered.imag)


def make_fft_matrix(window_size: int) -> np.ndarray:
    indices = bit_reverse_order(window_size)
    matrix = np.empty((window_size, window_size), dtype=np.complex128)
    for column in range(window_size):
        basis = np.zeros(window_size, dtype=np.float64)
        basis[column] = 1.0
        scrambled = fft_dif_radix2(basis)
        ordered = np.empty_like(scrambled)
        ordered[indices] = scrambled
        matrix[:, column] = ordered / window_size
    return matrix


def windows_to_features(windows: np.ndarray, fft_matrix: np.ndarray) -> np.ndarray:
    feature_blocks = []
    for channel_index in range(windows.shape[2]):
        channel_float = windows[:, :, channel_index].astype(np.float64) / Q15_SCALE
        fft_values = channel_float @ fft_matrix.T
        real_q15 = quantize_q15(fft_values.real)[:, FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(
            np.int32
        )
        imag_q15 = quantize_q15(fft_values.imag)[:, FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(
            np.int32
        )
        magnitude = np.abs(real_q15) + np.abs(imag_q15)
        feature_blocks.append(np.minimum(magnitude, Q15_MAX).astype(np.int16))
    return np.concatenate(feature_blocks, axis=1)


def window_to_features(window: np.ndarray) -> np.ndarray:
    channel_features = []
    for channel_index in range(window.shape[1]):
        real_q15, imag_q15 = fft_q15(window[:, channel_index])
        real_q15 = real_q15[FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(np.int32)
        imag_q15 = imag_q15[FIRST_FEATURE_BIN : LAST_FEATURE_BIN + 1].astype(np.int32)
        magnitude = np.abs(real_q15) + np.abs(imag_q15)
        channel_features.append(np.minimum(magnitude, Q15_MAX).astype(np.int16))
    return np.concatenate(channel_features)


def append_windows_from_run(
    run_values: np.ndarray,
    label: str,
    window_size: int,
    max_windows_per_class: int,
    fft_matrix: np.ndarray,
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
        features_by_class[class_id].extend(windows_to_features(windows, fft_matrix))

    consumed = window_count * window_size
    return run_values[consumed:]


def collect_balanced_windows(
    dataset: Path,
    window_size: int,
    max_windows_per_class: int,
    batch_size: int,
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

    return np.asarray(features, dtype=np.int16), np.asarray(labels, dtype=np.int8)


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
                    "feature_type": "approx_magnitude_abs_real_plus_abs_imag_q15",
                }
            )
            feature_index += 1
    return rows


def export_feature_map(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_tree_q15(model: DecisionTreeClassifier, path: Path) -> None:
    tree = model.tree_
    nodes = []
    for node_id in range(tree.node_count):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        values = tree.value[node_id][0]
        predicted_class = int(model.classes_[np.argmax(values)])

        if left == right:
            nodes.append(
                {
                    "node_id": node_id,
                    "is_leaf": True,
                    "class_id": predicted_class,
                    "class_label": ID_TO_LABEL[predicted_class],
                }
            )
        else:
            threshold = int(np.rint(tree.threshold[node_id]))
            nodes.append(
                {
                    "node_id": node_id,
                    "is_leaf": False,
                    "feature_index": int(tree.feature[node_id]),
                    "threshold_q15": int(np.clip(threshold, 0, Q15_MAX)),
                    "left_child": left,
                    "right_child": right,
                }
            )

    payload = {
        "q_format": "q1_15",
        "output_format": "class_id_integer_0_to_3",
        "class_map": {str(class_id): label for class_id, label in ID_TO_LABEL.items()},
        "decision_rule": "if feature[feature_index] <= threshold_q15 then left_child else right_child",
        "nodes": nodes,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def class_counts(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(class_id): int(counts.get(class_id, 0)) for class_id in ID_TO_LABEL}


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train(args: argparse.Namespace) -> dict[str, Any]:
    require_power_of_two(args.window_size)
    dataset = resolve_dataset(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = collect_balanced_windows(
        dataset,
        args.window_size,
        args.max_windows_per_class,
        args.batch_size,
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
        "window_size": args.window_size,
        "selected_fft_bins": {
            "first": FIRST_FEATURE_BIN,
            "last": LAST_FEATURE_BIN,
            "count_per_channel": FEATURE_BIN_COUNT,
        },
        "feature_count": int(features.shape[1]),
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
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Wrote artifacts to {output_dir}")
    return metrics


def write_comparison_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    best = max(
        rows,
        key=lambda row: (
            row["test_accuracy"],
            row["validation_accuracy"],
            row["cv_accuracy_mean"],
        ),
    )
    payload = {
        "selection_rule": (
            "best dataset is selected by highest test_accuracy, then "
            "validation_accuracy, then cv_accuracy_mean"
        ),
        "best_dataset": best["dataset"],
        "results": rows,
    }
    (output_root / "comparison_q15.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_root / "comparison_q15.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "artifact_dir",
            "test_accuracy",
            "validation_accuracy",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "tree_depth",
            "node_count",
            "feature_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_q15_datasets(args: argparse.Namespace) -> None:
    datasets = valid_q15_datasets()
    if not datasets:
        raise ValueError(f"No valid Q1.15 Parquet datasets found in {PROCESSED_DATASET_DIR}")

    output_root = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_ROOT
    rows = []
    for dataset in datasets:
        print(f"\n=== Training {dataset.name} ===")
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        dataset_args.output_dir = output_root / dataset.stem
        metrics = train(dataset_args)
        rows.append(
            {
                "dataset": dataset.name,
                "artifact_dir": metrics["artifact_dir"],
                "test_accuracy": metrics["test"]["accuracy"],
                "validation_accuracy": metrics["validation"]["accuracy"],
                "cv_accuracy_mean": metrics["cross_validation"]["accuracy_mean"],
                "cv_accuracy_std": metrics["cross_validation"]["accuracy_std"],
                "tree_depth": metrics["model"]["depth"],
                "node_count": metrics["model"]["node_count"],
                "feature_count": metrics["feature_count"],
            }
        )

    write_comparison_summary(rows, output_root)
    best = max(
        rows,
        key=lambda row: (
            row["test_accuracy"],
            row["validation_accuracy"],
            row["cv_accuracy_mean"],
        ),
    )
    print("\n=== Q1.15 comparison ===")
    for row in rows:
        print(
            f"{row['dataset']}: test={row['test_accuracy']:.4f}, "
            f"validation={row['validation_accuracy']:.4f}, "
            f"cv_mean={row['cv_accuracy_mean']:.4f}"
        )
    print(f"Best dataset: {best['dataset']} (test={best['test_accuracy']:.4f})")
    print(f"Wrote comparison to {output_root / 'comparison_q15.json'}")


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        list_datasets()
        return
    if args.compare_q15_datasets:
        compare_q15_datasets(args)
        return
    train(args)


if __name__ == "__main__":
    main()
