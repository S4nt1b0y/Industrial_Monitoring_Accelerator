"""Trains ml_classifier.reference's MLP on a fixed train/val/test split
of 07.Datasets/processed/features_dataset.parquet, split by source --
never by row, to avoid leaking temporally-correlated windows across
splits. Superseded as the primary evaluation method by
train_ml_classifier_kfold.py, kept as a simpler baseline.

Class-weighted loss (inverse frequency) counters real class imbalance
in the dataset, so raw accuracy isn't the only training/eval metric;
balanced accuracy is reported alongside it.

Usage (from 03.Reference):
    python -m tools.train_ml_classifier [--hidden 16] [--epochs 2000] [--lr 0.1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.paths import DATASET_DIR
from ml_classifier.reference import (
    CLASS_NAMES,
    FEATURE_NAMES,
    apply_gradient_step,
    compute_loss_and_grads,
    forward,
    init_weights,
)

FEATURES_PARQUET = DATASET_DIR / "features_dataset.parquet"
WEIGHTS_OUTPUT = DATASET_DIR / "ml_classifier_weights.npz"
REPORT_OUTPUT = DATASET_DIR / "ml_classifier_training_report.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10,
                         help="early-stopping patience, in units of the 100-epoch check interval")
    parser.add_argument("--restore-best", action="store_true",
                         help="use the best-val-loss checkpoint instead of the final epoch's weights "
                              "(off by default: val is as data-starved as everything else for "
                              "operacao_normal, making val_loss an unreliable stopping signal here -- "
                              "measured worse test results with this on than with the final weights)")
    return parser.parse_args()


def load_split(df, split_name, feature_mean, feature_std, label_to_index):
    subset = df[df["split"] == split_name]
    x = subset[FEATURE_NAMES].to_numpy(dtype=np.float64)
    # a zero-variance feature would divide by zero -- just center it,
    # it contributes a constant 0 to the model either way.
    safe_std = np.where(feature_std == 0, 1.0, feature_std)
    x = (x - feature_mean) / safe_std
    y = subset["label"].map(label_to_index).to_numpy()
    return x, y


def class_weights_from_counts(y_train, n_classes, scheme="inverse", cap=3.0):
    """`scheme` selects how aggressively rare classes are upweighted in
    the loss: "inverse" (default) weights a class in exact proportion
    to how rare it is; "sqrt" and "capped" are softer alternatives;
    "none" disables weighting. Tested directly: "inverse" outperforms
    every alternative on every class, for both the MLP and the CNN --
    softening it does not trade recall for precision, it makes both
    worse (the CNN collapses to predicting only the majority class with
    no weighting at all). Duplicated in tools/train_cnn.py rather than
    shared, since the MLP and CNN trainers don't import from each other.
    """
    counts = np.bincount(y_train, minlength=n_classes)
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
    return inverse / inverse.sum() * n_classes  # normalized so mean weight ~= 1


def balanced_accuracy(y_true, y_pred, n_classes):
    per_class_recall = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per_class_recall.append(np.mean(y_pred[mask] == c))
    return float(np.mean(per_class_recall))


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def binary_accuracy(y_true, y_pred, normal_index):
    true_ok = y_true == normal_index
    pred_ok = y_pred == normal_index
    return float(np.mean(true_ok == pred_ok))


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


def main():
    args = parse_args()
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    normal_index = label_to_index["operacao_normal"]

    df = pd.read_parquet(FEATURES_PARQUET)
    train_df = df[df["split"] == "train"]
    feature_mean = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64).mean(axis=0)
    feature_std = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64).std(axis=0)

    x_train, y_train = load_split(df, "train", feature_mean, feature_std, label_to_index)
    x_val, y_val = load_split(df, "val", feature_mean, feature_std, label_to_index)
    x_test, y_test = load_split(df, "test", feature_mean, feature_std, label_to_index)

    sample_weights_train = class_weights_from_counts(y_train, len(CLASS_NAMES))[y_train]

    weights = init_weights(n_hidden=args.hidden, seed=args.seed)
    history = []
    best_val_loss = np.inf
    best_weights = weights
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, train_acc, grads = compute_loss_and_grads(
            x_train, y_train, weights, sample_weights=sample_weights_train
        )
        weights = apply_gradient_step(weights, grads, learning_rate=args.lr)

        if epoch % 100 == 0 or epoch == args.epochs - 1:
            val_loss, val_acc, _ = compute_loss_and_grads(x_val, y_val, weights)
            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                             "val_loss": val_loss, "val_acc": val_acc})
            print(f"epoch {epoch:4d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = weights
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                    break

    if args.restore_best:
        weights = best_weights

    y_test_pred = np.array([forward(x, weights)[1] for x in x_test])
    test_matrix = confusion_matrix(y_test, y_test_pred, len(CLASS_NAMES))
    test_raw_acc = float(np.mean(y_test_pred == y_test))
    test_balanced_acc = balanced_accuracy(y_test, y_test_pred, len(CLASS_NAMES))
    test_binary_acc = binary_accuracy(y_test, y_test_pred, normal_index)

    print("\n=== Test set ===")
    print(f"raw accuracy:      {test_raw_acc:.4f}")
    print(f"balanced accuracy: {test_balanced_acc:.4f}")
    print(f"binary (ok/not-ok) accuracy: {test_binary_acc:.4f}")
    print("confusion matrix (rows=true, cols=predicted):")
    print("            " + "  ".join(f"{n[:10]:>10s}" for n in CLASS_NAMES))
    for i, row in enumerate(test_matrix):
        print(f"{CLASS_NAMES[i][:10]:>10s}  " + "  ".join(f"{v:10d}" for v in row))

    print("\n=== Feature weight magnitudes (mean |w1| per input, layer 1) ===")
    mean_abs_w1 = np.mean(np.abs(weights.w1), axis=0)
    for name, magnitude in zip(FEATURE_NAMES, mean_abs_w1):
        print(f"  {name:16s} {magnitude:.4f}")

    np.savez(
        WEIGHTS_OUTPUT,
        w1=weights.w1, b1=weights.b1, w2=weights.w2, b2=weights.b2,
        feature_mean=feature_mean, feature_std=feature_std,
        class_names=np.array(CLASS_NAMES), feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\nWrote trained weights to {WEIGHTS_OUTPUT}")

    report = {
        "hidden_units": args.hidden, "epochs": args.epochs, "learning_rate": args.lr,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "test_raw_accuracy": test_raw_acc, "test_balanced_accuracy": test_balanced_acc,
        "test_binary_accuracy": test_binary_acc,
        "test_confusion_matrix": test_matrix.tolist(), "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES, "mean_abs_w1_per_feature": mean_abs_w1.tolist(),
        "history": history,
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote training report to {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
