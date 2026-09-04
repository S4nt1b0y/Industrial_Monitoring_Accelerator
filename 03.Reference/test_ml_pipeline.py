#!/usr/bin/env python3
"""Unit tests for MLPipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_pipeline import FFT_FEATURE_COUNT, MDC_FEATURE_COUNT, MLPipeline, WINDOW_SIZE


def synthetic_channels(
    *,
    data_width: int = 16,
    windows_per_class: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = 16 if data_width == 8 else 1024
    channels = [[] for _ in range(4)]
    labels = []

    for class_id in range(4):
        for _ in range(windows_per_class):
            labels.append(class_id)
            for sample_index in range(WINDOW_SIZE):
                for channel_index, channel in enumerate(channels):
                    value = class_id * scale + channel_index * 2 + (sample_index % 4)
                    channel.append(value)

    return (
        np.asarray(channels[0], dtype=np.int16),
        np.asarray(channels[1], dtype=np.int16),
        np.asarray(channels[2], dtype=np.int16),
        np.asarray(channels[3], dtype=np.int16),
        np.asarray(labels, dtype=np.int8),
    )


class TestMLPipeline(unittest.TestCase):
    def test_initializes_supported_widths(self) -> None:
        self.assertEqual(MLPipeline().data_width, 8)
        self.assertEqual(MLPipeline(data_width=16).data_width, 16)
        self.assertEqual(MLPipeline(data_width=8).data_width, 8)

    def test_q17_defaults_to_no_lms_with_mdc(self) -> None:
        pipeline = MLPipeline(data_width=8)
        self.assertEqual(pipeline.q_format, "q1_7")
        self.assertEqual(pipeline.scale, 128.0)
        self.assertEqual(pipeline.min_sample_value, -128)
        self.assertEqual(pipeline.max_sample_value, 127)
        self.assertEqual(pipeline.max_feature_value, 127)
        self.assertFalse(pipeline.lms)
        self.assertTrue(pipeline.mdc)
        self.assertEqual(pipeline.n_features, FFT_FEATURE_COUNT + MDC_FEATURE_COUNT)

    def test_rejects_invalid_width(self) -> None:
        with self.assertRaises(ValueError):
            MLPipeline(data_width=12)

    def test_feature_count_without_and_with_mdc(self) -> None:
        self.assertEqual(MLPipeline(data_width=16, mdc=False).n_features, 132)
        self.assertEqual(MLPipeline(data_width=16, mdc=True).n_features, 140)

    def test_q17_fft_and_mdc_features_are_saturated(self) -> None:
        ch0, ch1, ch2, ch3, _ = synthetic_channels(data_width=8, windows_per_class=1)
        pipeline = MLPipeline(data_width=8)
        window = np.column_stack(
            [ch0[:WINDOW_SIZE], ch1[:WINDOW_SIZE], ch2[:WINDOW_SIZE], ch3[:WINDOW_SIZE]]
        )
        features = pipeline.window_to_features(window)
        fft_features = features[:FFT_FEATURE_COUNT]
        mdc_features = features[FFT_FEATURE_COUNT:]

        self.assertEqual(features.shape, (FFT_FEATURE_COUNT + MDC_FEATURE_COUNT,))
        self.assertEqual(mdc_features.shape, (MDC_FEATURE_COUNT,))
        self.assertTrue(np.all(fft_features >= 0))
        self.assertTrue(np.all(fft_features <= 127))
        self.assertTrue(np.all(mdc_features >= 0))
        self.assertTrue(np.all(mdc_features <= 127))

    def test_batch_feature_extraction_matches_scalar_path(self) -> None:
        ch0, ch1, ch2, ch3, _ = synthetic_channels(data_width=8, windows_per_class=2)
        pipeline = MLPipeline(data_width=8)
        batch = pipeline.channels_to_features((ch0, ch1, ch2, ch3))
        windows = np.column_stack([ch0, ch1, ch2, ch3]).reshape(-1, WINDOW_SIZE, 4)
        scalar = np.vstack([pipeline.window_to_features(window) for window in windows])

        np.testing.assert_array_equal(batch, scalar)

    def test_trains_with_synthetic_channels(self) -> None:
        ch0, ch1, ch2, ch3, labels = synthetic_channels()
        pipeline = MLPipeline(data_width=16, lms=False, mdc=False, min_samples_leaf=1)
        metrics = pipeline.train(ch0, ch1, ch2, ch3, labels, cv_folds=3)

        self.assertEqual(metrics["feature_count"], FFT_FEATURE_COUNT)
        self.assertEqual(metrics["pipeline"]["n_features"], 132)
        self.assertEqual(metrics["pipeline"]["lms"], False)
        self.assertEqual(metrics["pipeline"]["mdc"], False)

    def test_rejects_channels_with_different_lengths(self) -> None:
        ch0, ch1, ch2, ch3, labels = synthetic_channels()
        pipeline = MLPipeline(data_width=16, lms=False, min_samples_leaf=1)

        with self.assertRaises(ValueError):
            pipeline.train(ch0[:-1], ch1, ch2, ch3, labels, cv_folds=3)

    def test_rejects_wrong_label_count(self) -> None:
        ch0, ch1, ch2, ch3, labels = synthetic_channels()
        pipeline = MLPipeline(data_width=16, lms=False, min_samples_leaf=1)

        with self.assertRaises(ValueError):
            pipeline.train(ch0, ch1, ch2, ch3, labels[:-1], cv_folds=3)

    def test_streaming_classifier_returns_valid_on_complete_window(self) -> None:
        ch0, ch1, ch2, ch3, labels = synthetic_channels()
        pipeline = MLPipeline(data_width=16, lms=False, mdc=False, min_samples_leaf=1)
        pipeline.train(ch0, ch1, ch2, ch3, labels, cv_folds=3)

        for sample_index in range(WINDOW_SIZE - 1):
            valid, class_id = pipeline.classifier(
                int(ch0[sample_index]),
                int(ch1[sample_index]),
                int(ch2[sample_index]),
                int(ch3[sample_index]),
            )
            self.assertFalse(valid)
            self.assertIsNone(class_id)

        valid, class_id = pipeline.classifier(
            int(ch0[WINDOW_SIZE - 1]),
            int(ch1[WINDOW_SIZE - 1]),
            int(ch2[WINDOW_SIZE - 1]),
            int(ch3[WINDOW_SIZE - 1]),
        )
        self.assertTrue(valid)
        self.assertIsInstance(class_id, int)

    def test_exports_pipeline_config(self) -> None:
        ch0, ch1, ch2, ch3, labels = synthetic_channels(data_width=8)
        pipeline = MLPipeline(data_width=8, lms=False, mdc=True, min_samples_leaf=1)
        pipeline.train(ch0, ch1, ch2, ch3, labels, cv_folds=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pipeline.save_artifacts(output_dir)
            config_path = output_dir / "pipeline_config.json"

            self.assertTrue((output_dir / "model.joblib").is_file())
            self.assertTrue((output_dir / "tree_q1_7.json").is_file())
            self.assertTrue(config_path.is_file())

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["data_width"], 8)
            self.assertEqual(config["window_size"], WINDOW_SIZE)
            self.assertEqual(config["lms"], False)
            self.assertEqual(config["mdc"], True)
            self.assertEqual(config["n_features"], 140)


if __name__ == "__main__":
    unittest.main()
