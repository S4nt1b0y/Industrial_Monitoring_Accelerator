#!/usr/bin/env python3
"""Unit tests for MLClassifier."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_classifier import MLClassifier


def synthetic_dataset(data_width: int) -> tuple[np.ndarray, np.ndarray]:
    scale = 16 if data_width == 8 else 1024
    features = []
    labels = []
    for class_id in range(4):
        for offset in range(12):
            value = class_id * scale + offset
            features.append([value, value + 1, value + 2])
            labels.append(class_id)
    return np.asarray(features, dtype=np.int16), np.asarray(labels, dtype=np.int8)


class TestMLClassifier(unittest.TestCase):
    def test_initializes_supported_widths(self) -> None:
        self.assertEqual(MLClassifier(n_features=3, data_width=16).q_format, "q1_15")
        self.assertEqual(MLClassifier(n_features=3, data_width=8).q_format, "q1_7")

    def test_rejects_invalid_width(self) -> None:
        with self.assertRaises(ValueError):
            MLClassifier(n_features=3, data_width=12)

    def test_rejects_wrong_feature_count(self) -> None:
        classifier = MLClassifier(n_features=3, data_width=16)
        features = np.zeros((4, 2), dtype=np.int16)
        labels = np.zeros(4, dtype=np.int8)
        with self.assertRaises(ValueError):
            classifier.train(features, labels, cv_folds=2)

    def test_rejects_out_of_range_features(self) -> None:
        classifier = MLClassifier(n_features=3, data_width=8)
        features = np.asarray([[0, 1, 128]], dtype=np.int16)
        labels = np.asarray([0], dtype=np.int8)
        with self.assertRaises(ValueError):
            classifier.train(features, labels, cv_folds=2)

    def test_rejects_classification_before_training(self) -> None:
        classifier = MLClassifier(n_features=3, data_width=16)
        with self.assertRaises(RuntimeError):
            classifier.classify(np.zeros((1, 3), dtype=np.int16))

    def test_trains_and_classifies_synthetic_q15_dataset(self) -> None:
        features, labels = synthetic_dataset(data_width=16)
        classifier = MLClassifier(n_features=3, data_width=16, min_samples_leaf=1)
        metrics = classifier.train(features, labels, cv_folds=3)
        predictions = classifier.classify(features[:5])
        alias_predictions = classifier.classifier(features[:5])

        self.assertEqual(metrics["q_format"], "q1_15")
        self.assertEqual(metrics["feature_count"], 3)
        self.assertEqual(predictions.shape, (5,))
        np.testing.assert_array_equal(predictions, alias_predictions)

    def test_trains_q17_and_exports_artifacts(self) -> None:
        features, labels = synthetic_dataset(data_width=8)
        classifier = MLClassifier(n_features=3, data_width=8, min_samples_leaf=1)
        classifier.train(features, labels, cv_folds=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            classifier.save_artifacts(output_dir)
            tree_path = output_dir / "tree_q1_7.json"

            self.assertTrue((output_dir / "model.joblib").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue(tree_path.is_file())

            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            self.assertEqual(tree["q_format"], "q1_7")
            self.assertEqual(tree["data_width"], 8)
            internal_nodes = [node for node in tree["nodes"] if not node["is_leaf"]]
            self.assertTrue(internal_nodes)
            self.assertIn("threshold", internal_nodes[0])
            self.assertNotIn("threshold_q15", internal_nodes[0])


if __name__ == "__main__":
    unittest.main()
