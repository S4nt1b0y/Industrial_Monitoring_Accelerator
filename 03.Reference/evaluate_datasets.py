#!/usr/bin/env python3
"""Evaluate processed Parquet datasets with the fixed MLPipeline configuration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from ml_classifier import ID_TO_LABEL, LABEL_TO_ID
from ml_pipeline import CHANNEL_COUNT, MLPipeline, WINDOW_SIZE


DEFAULT_DATASET_DIR = Path("07.Datasets/processed")
DEFAULT_OUTPUT_DIR = Path("03.Reference/artifacts/dataset_evaluation")
VIBRATION_COLUMNS = (
    "aceleracao_x_mancal_a",
    "aceleracao_y_mancal_a",
    "aceleracao_x_mancal_b",
    "aceleracao_y_mancal_b",
)
PIPELINE_CONFIGS = (
    {"name": "lms_off_mdc_on", "lms": False, "mdc": True},
)


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    data_width: int
    rows: int


def infer_dataset_info(path: Path) -> DatasetInfo:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    names = set(schema.names)
    required = {"label", *VIBRATION_COLUMNS}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    channel_types = [schema.field(column).type for column in VIBRATION_COLUMNS]
    if all(pa.types.is_int16(channel_type) for channel_type in channel_types):
        data_width = 16
    elif all(pa.types.is_int8(channel_type) for channel_type in channel_types):
        data_width = 8
    else:
        raise ValueError(f"{path.name} has unsupported channel types: {channel_types}")

    return DatasetInfo(
        path=path,
        data_width=data_width,
        rows=parquet_file.metadata.num_rows,
    )


def discover_datasets(dataset_dir: Path) -> tuple[list[DatasetInfo], list[dict[str, str]]]:
    valid = []
    invalid = []
    for path in sorted(dataset_dir.glob("*.parquet")):
        try:
            valid.append(infer_dataset_info(path))
        except ValueError as exc:
            invalid.append({"dataset": path.name, "reason": str(exc)})
    return valid, invalid


def count_windows_by_class(path: Path, batch_size: int) -> dict[int, int]:
    counts: Counter[int] = Counter()
    current_label: str | None = None
    current_count = 0

    for labels in iter_label_batches(path, batch_size):
        start = 0
        while start < len(labels):
            label = str(labels[start])
            stop = start + 1
            while stop < len(labels) and labels[stop] == labels[start]:
                stop += 1

            if label not in LABEL_TO_ID:
                raise ValueError(f"unknown label in {path.name}: {label}")
            if current_label is None:
                current_label = label
                current_count = stop - start
            elif current_label == label:
                current_count += stop - start
            else:
                counts[LABEL_TO_ID[current_label]] += current_count // WINDOW_SIZE
                current_label = label
                current_count = stop - start
            start = stop

    if current_label is not None:
        counts[LABEL_TO_ID[current_label]] += current_count // WINDOW_SIZE

    return {class_id: int(counts.get(class_id, 0)) for class_id in ID_TO_LABEL}


def iter_label_batches(path: Path, batch_size: int):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["label"]):
        yield batch.column("label").to_numpy(zero_copy_only=False)


def balanced_target(
    counts: dict[int, int],
    max_windows_per_class: int | None,
) -> int:
    target = min(counts.values()) if counts else 0
    if max_windows_per_class is not None:
        target = min(target, max_windows_per_class)
    if target <= 0:
        raise ValueError(f"not enough complete windows for every class: {counts}")
    return int(target)


def collect_balanced_windows(
    path: Path,
    target_per_class: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["label", *VIBRATION_COLUMNS]
    parquet_file = pq.ParquetFile(path)
    windows_by_class: dict[int, list[np.ndarray]] = {class_id: [] for class_id in ID_TO_LABEL}
    carry_label: str | None = None
    carry_values = np.empty((0, CHANNEL_COUNT), dtype=np.int32)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        table = batch
        labels = table.column("label").to_numpy(zero_copy_only=False)
        values = np.column_stack(
            [
                table.column(column).to_numpy(zero_copy_only=False)
                for column in VIBRATION_COLUMNS
            ]
        ).astype(np.int32, copy=False)

        start = 0
        while start < len(labels):
            label = str(labels[start])
            stop = start + 1
            while stop < len(labels) and labels[stop] == labels[start]:
                stop += 1

            if label not in LABEL_TO_ID:
                raise ValueError(f"unknown label in {path.name}: {label}")

            run_values = values[start:stop]
            if carry_label == label and carry_values.size:
                run_values = np.vstack([carry_values, run_values])
            elif carry_label != label:
                carry_values = carry_values[:0]

            class_id = LABEL_TO_ID[label]
            run_values, carry_values = consume_run_windows(
                run_values,
                windows_by_class[class_id],
                target_per_class,
            )
            carry_label = label
            start = stop

        if all(len(items) >= target_per_class for items in windows_by_class.values()):
            break

    missing = {
        ID_TO_LABEL[class_id]: target_per_class - len(items)
        for class_id, items in windows_by_class.items()
        if len(items) < target_per_class
    }
    if missing:
        raise ValueError(f"could not collect balanced windows from {path.name}: {missing}")

    windows = []
    labels = []
    for class_id in ID_TO_LABEL:
        selected = windows_by_class[class_id][:target_per_class]
        windows.extend(selected)
        labels.extend([class_id] * len(selected))

    return np.asarray(windows, dtype=np.int32), np.asarray(labels, dtype=np.int8)


def consume_run_windows(
    run_values: np.ndarray,
    class_windows: list[np.ndarray],
    target_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = target_per_class - len(class_windows)
    if remaining <= 0:
        return run_values[:0], run_values[:0]

    complete_windows = min(run_values.shape[0] // WINDOW_SIZE, remaining)
    if complete_windows:
        stop = complete_windows * WINDOW_SIZE
        class_windows.extend(
            run_values[:stop].reshape(complete_windows, WINDOW_SIZE, CHANNEL_COUNT)
        )
        run_values = run_values[stop:]

    return run_values, run_values


def split_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_windows, test_windows, train_labels, test_labels = train_test_split(
        windows,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_windows, test_windows, train_labels, test_labels


def windows_to_channels(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flattened = windows.reshape(windows.shape[0] * WINDOW_SIZE, CHANNEL_COUNT)
    return tuple(flattened[:, channel_index] for channel_index in range(CHANNEL_COUNT))  # type: ignore[return-value]


def streaming_metrics(
    pipeline: MLPipeline,
    windows: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    predictions = []
    expected = []
    invalid_outputs = 0

    for window, label in zip(windows, labels):
        for sample_index in range(WINDOW_SIZE):
            valid, class_id = pipeline.classifier(
                int(window[sample_index, 0]),
                int(window[sample_index, 1]),
                int(window[sample_index, 2]),
                int(window[sample_index, 3]),
            )
            if sample_index < WINDOW_SIZE - 1 and valid:
                invalid_outputs += 1
            if sample_index == WINDOW_SIZE - 1:
                if valid and class_id is not None:
                    predictions.append(class_id)
                    expected.append(int(label))
                else:
                    invalid_outputs += 1

    accuracy = float(accuracy_score(expected, predictions)) if predictions else 0.0
    return {
        "accuracy": accuracy,
        "valid_predictions": len(predictions),
        "expected_predictions": int(labels.shape[0]),
        "invalid_outputs": invalid_outputs,
        "confusion_matrix": confusion_matrix(
            expected,
            predictions,
            labels=list(ID_TO_LABEL),
        ).tolist(),
    }


def evaluate_dataset(
    info: DatasetInfo,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    counts = count_windows_by_class(info.path, args.batch_size)
    target_per_class = balanced_target(counts, args.max_windows_per_class)
    windows, labels = collect_balanced_windows(info.path, target_per_class, args.batch_size)
    train_windows, test_windows, train_labels, test_labels = split_windows(
        windows,
        labels,
        args.test_size,
        args.seed,
    )
    train_channels = windows_to_channels(train_windows)
    results = []

    for config in PIPELINE_CONFIGS:
        print(f"Training {info.path.name} / {config['name']}")
        pipeline = MLPipeline(
            data_width=info.data_width,
            lms=bool(config["lms"]),
            mdc=bool(config["mdc"]),
            fs_hz=args.fs_hz,
            min_k=args.min_k,
            lms_delay=args.lms_delay,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )
        train_metrics = pipeline.train(
            *train_channels,
            train_labels,
            test_size=args.internal_test_size,
            val_size=args.val_size,
            cv_folds=args.cv_folds,
        )
        stream = streaming_metrics(pipeline, test_windows, test_labels)
        output_dir = args.output_dir / info.path.stem / str(config["name"])
        pipeline.save_artifacts(output_dir)

        result = {
            "dataset": {
                "name": info.path.name,
                "path": str(info.path),
                "rows": info.rows,
                "data_width": info.data_width,
            },
            "configuration": config,
            "counts": {
                "available_windows_by_class": {
                    str(class_id): counts[class_id] for class_id in ID_TO_LABEL
                },
                "balanced_windows_per_class": target_per_class,
                "total_balanced_windows": int(labels.shape[0]),
                "train_windows": int(train_labels.shape[0]),
                "test_windows": int(test_labels.shape[0]),
            },
            "train_metrics": train_metrics,
            "streaming_metrics": stream,
            "artifact_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    return results


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    train_metrics = result["train_metrics"]
    stream = result["streaming_metrics"]
    config = result["configuration"]
    return {
        "dataset": result["dataset"]["name"],
        "data_width": result["dataset"]["data_width"],
        "lms": config["lms"],
        "mdc": config["mdc"],
        "n_features": train_metrics["pipeline"]["n_features"],
        "balanced_windows_per_class": result["counts"]["balanced_windows_per_class"],
        "internal_train_accuracy": train_metrics["train"]["accuracy"],
        "internal_validation_accuracy": train_metrics["validation"]["accuracy"],
        "internal_test_accuracy": train_metrics["test"]["accuracy"],
        "streaming_accuracy": stream["accuracy"],
        "streaming_valid_predictions": stream["valid_predictions"],
        "artifact_dir": result["artifact_dir"],
    }


def write_comparison(
    output_dir: Path,
    results: list[dict[str, Any]],
    invalid_datasets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_result(result) for result in results]
    rows.sort(
        key=lambda row: (
            row["streaming_accuracy"],
            row["internal_test_accuracy"],
        ),
        reverse=True,
    )
    payload = {
        "results": results,
        "summary": rows,
        "invalid_datasets": invalid_datasets,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if rows:
        with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, Any]], invalid_datasets: list[dict[str, str]]) -> None:
    if invalid_datasets:
        print("\nInvalid datasets:")
        for item in invalid_datasets:
            print(f"- {item['dataset']}: {item['reason']}")

    if not rows:
        print("\nNo valid evaluation results.")
        return

    print("\nEvaluation summary:")
    header = (
        "dataset",
        "width",
        "lms",
        "mdc",
        "features",
        "balanced/class",
        "train",
        "test",
        "stream",
    )
    print(
        f"{header[0]:32} {header[1]:>5} {header[2]:>5} {header[3]:>5} "
        f"{header[4]:>8} {header[5]:>14} {header[6]:>8} {header[7]:>8} {header[8]:>8}"
    )
    for row in rows:
        print(
            f"{row['dataset'][:32]:32} {row['data_width']:>5} "
            f"{str(row['lms']):>5} {str(row['mdc']):>5} "
            f"{row['n_features']:>8} {row['balanced_windows_per_class']:>14} "
            f"{row['internal_train_accuracy']:>8.4f} "
            f"{row['internal_test_accuracy']:>8.4f} "
            f"{row['streaming_accuracy']:>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every processed Parquet dataset with LMS off and MDC on."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--internal-test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--max-windows-per-class", type=int, default=None)
    parser.add_argument("--fs-hz", type=int, default=6400)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--lms-delay", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, invalid = discover_datasets(args.dataset_dir)
    results: list[dict[str, Any]] = []

    if not datasets:
        print(f"No valid Parquet datasets found in {args.dataset_dir}")

    for dataset in datasets:
        try:
            results.extend(evaluate_dataset(dataset, args))
        except Exception as exc:
            invalid.append({"dataset": dataset.path.name, "reason": str(exc)})

    rows = write_comparison(args.output_dir, results, invalid)
    print_summary(rows, invalid)


if __name__ == "__main__":
    main()
