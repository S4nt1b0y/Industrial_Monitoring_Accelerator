#!/usr/bin/env python3
"""Unit tests for evaluate_datasets.py."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_datasets import (
    PIPELINE_CONFIGS,
    VIBRATION_COLUMNS,
    balanced_target,
    collect_balanced_windows,
    count_windows_by_class,
    discover_datasets,
    evaluate_dataset,
    infer_dataset_info,
    write_comparison,
)
from ml_classifier import ID_TO_LABEL
from ml_pipeline import WINDOW_SIZE


def write_vibration_parquet(
    path: Path,
    windows_by_class: dict[int, int],
    *,
    data_width: int = 16,
) -> None:
    labels = []
    channels = {column: [] for column in VIBRATION_COLUMNS}
    dtype = np.int16 if data_width == 16 else np.int8
    scale = 16 if data_width == 8 else 1024

    for class_id, window_count in windows_by_class.items():
        label = ID_TO_LABEL[class_id]
        for _ in range(window_count):
            for sample_index in range(WINDOW_SIZE):
                labels.append(label)
                for channel_index, column in enumerate(VIBRATION_COLUMNS):
                    value = class_id * scale + channel_index * 2 + (sample_index % 4)
                    channels[column].append(value)

    arrays = {"label": pa.array(labels, type=pa.large_string())}
    for column, values in channels.items():
        arrays[column] = pa.array(np.asarray(values, dtype=dtype))
    pq.write_table(pa.table(arrays), path)


class TestEvaluateDatasets(unittest.TestCase):
    def test_infers_data_width_from_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            q15_path = Path(tmpdir) / "q15.parquet"
            q17_path = Path(tmpdir) / "q17.parquet"
            write_vibration_parquet(q15_path, {0: 1, 1: 1, 2: 1, 3: 1}, data_width=16)
            write_vibration_parquet(q17_path, {0: 1, 1: 1, 2: 1, 3: 1}, data_width=8)

            self.assertEqual(infer_dataset_info(q15_path).data_width, 16)
            self.assertEqual(infer_dataset_info(q17_path).data_width, 8)

    def test_discovers_invalid_dataset_missing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            valid_path = dataset_dir / "valid.parquet"
            invalid_path = dataset_dir / "invalid.parquet"
            write_vibration_parquet(valid_path, {0: 1, 1: 1, 2: 1, 3: 1})
            pq.write_table(
                pa.table({"label": pa.array(["operacao_normal"], type=pa.large_string())}),
                invalid_path,
            )

            valid, invalid = discover_datasets(dataset_dir)
            self.assertEqual([item.path.name for item in valid], ["valid.parquet"])
            self.assertEqual(invalid[0]["dataset"], "invalid.parquet")

    def test_counts_windows_without_crossing_label_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runs.parquet"
            labels = (
                ["operacao_normal"] * 32
                + ["desalinhamento"] * 64
                + ["operacao_normal"] * 32
            )
            arrays = {"label": pa.array(labels, type=pa.large_string())}
            for column in VIBRATION_COLUMNS:
                arrays[column] = pa.array(np.zeros(len(labels), dtype=np.int16))
            pq.write_table(pa.table(arrays), path)

            counts = count_windows_by_class(path, batch_size=40)
            self.assertEqual(counts[0], 0)
            self.assertEqual(counts[1], 1)

    def test_collects_balanced_windows_by_smallest_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "balanced.parquet"
            counts = {0: 2, 1: 3, 2: 4, 3: 5}
            write_vibration_parquet(path, counts)
            available = count_windows_by_class(path, batch_size=100)
            target = balanced_target(available, max_windows_per_class=None)
            windows, labels = collect_balanced_windows(path, target, batch_size=100)

            self.assertEqual(target, 2)
            self.assertEqual(windows.shape, (8, WINDOW_SIZE, 4))
            self.assertEqual(labels.tolist().count(0), 2)
            self.assertEqual(labels.tolist().count(3), 2)

    def test_evaluates_all_pipeline_configs_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset_q15.parquet"
            output_dir = root / "artifacts"
            write_vibration_parquet(dataset_path, {0: 12, 1: 12, 2: 12, 3: 12})
            info = infer_dataset_info(dataset_path)
            args = argparse.Namespace(
                batch_size=100,
                max_windows_per_class=None,
                test_size=0.25,
                internal_test_size=0.15,
                val_size=0.15,
                cv_folds=2,
                seed=42,
                fs_hz=6400,
                min_k=2,
                lms_delay=1,
                max_depth=5,
                min_samples_leaf=1,
                output_dir=output_dir,
            )

            results = evaluate_dataset(info, args)
            rows = write_comparison(output_dir, results, [])

            self.assertEqual(len(results), len(PIPELINE_CONFIGS))
            self.assertEqual(len(rows), len(PIPELINE_CONFIGS))
            self.assertTrue((output_dir / "comparison.json").is_file())
            self.assertTrue((output_dir / "comparison.csv").is_file())
            self.assertTrue(
                (output_dir / "dataset_q15" / "lms_on_mdc_off" / "result.json").is_file()
            )

            comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["summary"][0]["balanced_windows_per_class"], 12)


if __name__ == "__main__":
    unittest.main()
