#!/usr/bin/env python3
"""Reusable RTL-friendly motor-state classifier.

This module intentionally does not load datasets, build windows, or extract FFT
features. A top-level script is responsible for preparing fixed-point feature
matrices and labels before calling MLClassifier.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


DEFAULT_OUTPUT_ROOT = Path("03.Reference/artifacts/ml_classifier")
LABEL_TO_ID = {
    "operacao_normal": 0,
    "desalinhamento": 1,
    "desbalanceamento": 2,
    "desgaste_rolamento": 3,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SUPPORTED_DATA_WIDTHS = (8, 16)
MAX_FEATURE_BY_WIDTH = {
    8: 127,
    16: 32767,
}


def q_format(data_width: int) -> str:
    if data_width not in SUPPORTED_DATA_WIDTHS:
        raise ValueError(f"data_width must be one of {SUPPORTED_DATA_WIDTHS}, got {data_width}")
    return f"q1_{data_width - 1}"


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


class MLClassifier:
    """Decision-tree classifier for fixed-point features prepared by a top layer."""

    def __init__(
        self,
        n_features: int,
        data_width: int,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 16,
        class_weight: str | dict[int, float] | None = "balanced",
        seed: int = 42,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if data_width not in SUPPORTED_DATA_WIDTHS:
            raise ValueError(f"data_width must be one of {SUPPORTED_DATA_WIDTHS}, got {data_width}")

        self.n_features = int(n_features)
        self.data_width = int(data_width)
        self.max_feature_value = MAX_FEATURE_BY_WIDTH[self.data_width]
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.seed = seed
        self.q_format = q_format(self.data_width)
        self.model: DecisionTreeClassifier | None = None
        self.metrics: dict[str, Any] | None = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        test_size: float = 0.15,
        val_size: float = 0.15,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        features = self._validate_features(features)
        labels = self._validate_labels(labels, expected_samples=features.shape[0])
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(
            features,
            labels,
            test_size,
            val_size,
            self.seed,
        )

        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.seed,
        )

        cv_scores = self._cross_validation_scores(model, x_train, y_train, cv_folds)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)

        self.model = model
        self.metrics = {
            "q_format": self.q_format,
            "data_width": self.data_width,
            "feature_count": self.n_features,
            "feature_range": {
                "min": 0,
                "max": self.max_feature_value,
            },
            "class_map": {str(class_id): label for class_id, label in ID_TO_LABEL.items()},
            "classifier_output": "class_id_integer_0_to_3",
            "sample_counts": {
                "total": class_counts(labels),
                "train": class_counts(y_train),
                "validation": class_counts(y_val),
                "test": class_counts(y_test),
            },
            "model": {
                "type": "DecisionTreeClassifier",
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "class_weight": self.class_weight,
                "node_count": int(model.tree_.node_count),
                "depth": int(model.tree_.max_depth),
            },
            "cross_validation": {
                "folds": cv_folds,
                "accuracy_scores": cv_scores.tolist(),
                "accuracy_mean": float(np.mean(cv_scores)) if len(cv_scores) else None,
                "accuracy_std": float(np.std(cv_scores)) if len(cv_scores) else None,
            },
            "train": self._evaluation(y_train, train_pred),
            "validation": self._evaluation(y_val, val_pred),
            "test": self._evaluation(y_test, test_pred),
        }
        return self.metrics

    def classify(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLClassifier must be trained before classify()")
        features = self._validate_features(features)
        return self.model.predict(features).astype(np.int8)

    def classifier(self, features: np.ndarray) -> np.ndarray:
        return self.classify(features)

    def save_artifacts(self, output_dir: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("MLClassifier must be trained before save_artifacts()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path / "model.joblib")
        (output_path / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.export_tree(output_path / f"tree_{self.q_format}.json")

    def export_tree(self, path: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("MLClassifier must be trained before export_tree()")

        tree = self.model.tree_
        nodes = []
        for node_id in range(tree.node_count):
            left = int(tree.children_left[node_id])
            right = int(tree.children_right[node_id])
            values = tree.value[node_id][0]
            predicted_class = int(self.model.classes_[np.argmax(values)])

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
                        "threshold": int(np.clip(threshold, 0, self.max_feature_value)),
                        "left_child": left,
                        "right_child": right,
                    }
                )

        payload = {
            "q_format": self.q_format,
            "data_width": self.data_width,
            "output_format": "class_id_integer_0_to_3",
            "class_map": {str(class_id): label for class_id, label in ID_TO_LABEL.items()},
            "decision_rule": "if feature[feature_index] <= threshold then left_child else right_child",
            "nodes": nodes,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError(f"features must be a 2-D array, got shape {features.shape}")
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"features must have {self.n_features} columns, got {features.shape[1]}"
            )
        if not np.issubdtype(features.dtype, np.integer):
            raise ValueError(f"features must contain integer fixed-point values, got {features.dtype}")
        if features.size:
            min_value = int(np.min(features))
            max_value = int(np.max(features))
            if min_value < 0 or max_value > self.max_feature_value:
                raise ValueError(
                    "features out of range for "
                    f"{self.q_format}: expected 0..{self.max_feature_value}, "
                    f"got {min_value}..{max_value}"
                )
        return features.astype(np.int32, copy=False)

    @staticmethod
    def _validate_labels(labels: np.ndarray, *, expected_samples: int) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"labels must be a 1-D array, got shape {labels.shape}")
        if labels.shape[0] != expected_samples:
            raise ValueError(
                f"labels must have {expected_samples} samples, got {labels.shape[0]}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels must contain integer class ids, got {labels.dtype}")

        valid_classes = set(ID_TO_LABEL)
        invalid = sorted(set(int(value) for value in labels) - valid_classes)
        if invalid:
            raise ValueError(f"labels contain unknown class ids: {invalid}")
        return labels.astype(np.int8, copy=False)

    def _cross_validation_scores(
        self,
        model: DecisionTreeClassifier,
        features: np.ndarray,
        labels: np.ndarray,
        cv_folds: int,
    ) -> np.ndarray:
        if cv_folds <= 1:
            return np.asarray([], dtype=np.float64)

        minimum_class_count = min(Counter(int(label) for label in labels).values())
        if cv_folds > minimum_class_count:
            raise ValueError(
                f"cv_folds={cv_folds} is greater than the smallest training class "
                f"count ({minimum_class_count})"
            )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        return cross_val_score(model, features, labels, cv=cv, scoring="accuracy")

    @staticmethod
    def _evaluation(expected: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(expected, predicted)),
            "confusion_matrix": confusion_matrix(
                expected,
                predicted,
                labels=list(ID_TO_LABEL),
            ).tolist(),
            "classification_report": classification_report(
                expected,
                predicted,
                labels=list(ID_TO_LABEL),
                target_names=[ID_TO_LABEL[index] for index in ID_TO_LABEL],
                output_dict=True,
                zero_division=0,
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train MLClassifier from precomputed fixed-point feature arrays. "
            "Datasets, windows and FFT features must be prepared by a top-level script."
        )
    )
    parser.add_argument("--features", type=Path, required=True, help="Input .npy feature matrix.")
    parser.add_argument("--labels", type=Path, required=True, help="Input .npy training labels.")
    parser.add_argument("--n-features", type=int, required=True)
    parser.add_argument("--data-width", type=int, choices=SUPPORTED_DATA_WIDTHS, required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def train_from_args(args: argparse.Namespace) -> dict[str, Any]:
    features = np.load(args.features)
    labels = np.load(args.labels)
    classifier = MLClassifier(
        n_features=args.n_features,
        data_width=args.data_width,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    metrics = classifier.train(
        features,
        labels,
        test_size=args.test_size,
        val_size=args.val_size,
        cv_folds=args.cv_folds,
    )
    classifier.save_artifacts(args.output_dir)

    print(f"Trained MLClassifier with {args.n_features} features in {classifier.q_format}")
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")
    return metrics


def main() -> None:
    train_from_args(parse_args())


if __name__ == "__main__":
    main()
