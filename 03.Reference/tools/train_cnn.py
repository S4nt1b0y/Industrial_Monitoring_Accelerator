"""Shared metrics and class-weighting helpers for the CNN training
scripts (tools/train_cnn_kfold.py). Kept in this module for parity with
tools/train_ml_classifier.py, which serves the same role for the MLP.
"""

import numpy as np


def class_weights_from_counts(y, n_classes, scheme="inverse", cap=3.0):
    """`scheme` selects how aggressively rare classes are upweighted in
    the loss: "inverse" (default) weights a class in exact proportion
    to how rare it is; "sqrt" and "capped" are softer alternatives;
    "none" disables weighting. Tested directly: "inverse" outperforms
    every alternative on every class -- softening it does not trade
    recall for precision, it makes both worse, and with no weighting at
    all the CNN collapses to predicting only the majority class.
    """
    counts = np.bincount(y, minlength=n_classes)
    if scheme == "none":
        return np.ones(n_classes)
    if scheme == "sqrt":
        inverse = 1.0 / np.sqrt(np.maximum(counts, 1))
    elif scheme == "capped":
        inverse = 1.0 / np.maximum(counts, 1)
        inverse = np.minimum(inverse, inverse.min() * cap)
    elif scheme == "inverse":
        inverse = 1.0 / np.maximum(counts, 1)
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    return inverse / inverse.sum() * n_classes


def balanced_accuracy(y_true, y_pred, n_classes):
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(np.mean(y_pred[mask] == c))
    return float(np.mean(recalls))


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def binary_accuracy(y_true, y_pred, normal_index):
    return float(np.mean((y_true == normal_index) == (y_pred == normal_index)))


def per_class_precision_recall_f1(matrix, class_names):
    matrix = np.asarray(matrix)
    result = {}
    for c, name in enumerate(class_names):
        tp = int(matrix[c, c])
        support = int(matrix[c].sum())
        predicted = int(matrix[:, c].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return result
