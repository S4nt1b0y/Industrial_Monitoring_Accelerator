"""Pipeline v2 counterpart to tools/train_ml_classifier_kfold.py -- same
training regime (grouped k-fold, class-weighted full-batch GD, optional
--save-final official-weights pass), but against ml_classifier.reference_v2
/ features_dataset_v2.parquet (v1's vector + 4 AR(4) coefficients).
Writes separate output files -- v1's official ml_classifier_weights.npz
is untouched.

Usage (from 03.Reference):
    python -m tools.train_ml_classifier_kfold_v2 [--hidden 16] [--epochs 2000] [--lr 0.1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.paths import DATASET_DIR
from ml_classifier.reference_v2 import (
    CLASS_NAMES,
    FEATURE_NAMES,
    apply_gradient_step,
    compute_loss_and_grads,
    forward,
    init_weights,
)
from tools.train_ml_classifier import (
    balanced_accuracy,
    binary_accuracy,
    class_weights_from_counts,
    confusion_matrix,
    per_class_precision_recall_f1,
)

FEATURES_PARQUET = DATASET_DIR / "features_dataset_v2.parquet"
REPORT_OUTPUT = DATASET_DIR / "ml_classifier_v2_kfold_report.json"
FINAL_WEIGHTS_OUTPUT = DATASET_DIR / "ml_classifier_v2_weights.npz"
FINAL_REPORT_OUTPUT = DATASET_DIR / "ml_classifier_v2_training_report.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--weighting", type=str, default="inverse",
                         choices=["inverse", "sqrt", "none", "capped"],
                         help="class-weighting scheme, see "
                              "tools/train_ml_classifier.class_weights_from_counts")
    parser.add_argument("--features", type=Path, default=FEATURES_PARQUET,
                         help="point this at features_dataset_v2_dense_normal.parquet "
                              "to train against the version with denser operacao_normal sampling")
    parser.add_argument("--save-final", action="store_true",
                         help="after the k-fold comparison, train one more model on ALL rows "
                              "(no held-out data) and save it as the official v2 "
                              "ml_classifier_v2_weights.npz/training_report.json")
    return parser.parse_args()


def train_one_fold(x_train, y_train, x_test, y_test, args, fold_seed):
    sample_weights_train = class_weights_from_counts(y_train, len(CLASS_NAMES), scheme=args.weighting)[y_train]
    weights = init_weights(n_hidden=args.hidden, seed=fold_seed)

    for epoch in range(args.epochs):
        _, _, grads = compute_loss_and_grads(x_train, y_train, weights, sample_weights=sample_weights_train)
        weights = apply_gradient_step(weights, grads, learning_rate=args.lr)

    y_pred = np.array([forward(x, weights)[1] for x in x_test])
    matrix = confusion_matrix(y_test, y_pred, len(CLASS_NAMES))
    raw_acc = float(np.mean(y_pred == y_test))
    bal_acc = balanced_accuracy(y_test, y_pred, len(CLASS_NAMES))
    return raw_acc, bal_acc, matrix, y_pred


def main():
    args = parse_args()
    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    normal_index = label_to_index["operacao_normal"]

    df = pd.read_parquet(args.features)
    n_classes = len(CLASS_NAMES)

    raw_accs, bal_accs, bin_accs = [], [], []
    total_matrix = np.zeros((n_classes, n_classes), dtype=int)

    print(f"pipeline v2 -- {len(FEATURE_NAMES)} features")
    print(f"{'fold':>5}{'n_train':>9}{'n_test':>8}{'raw_acc':>10}{'bal_acc':>10}{'bin_acc':>10}")
    for fold in range(args.n_folds):
        test_df = df[df["fold"] == fold]
        train_df = df[df["fold"] != fold]

        feature_mean = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64).mean(axis=0)
        feature_std = train_df[FEATURE_NAMES].to_numpy(dtype=np.float64).std(axis=0)
        safe_std = np.where(feature_std == 0, 1.0, feature_std)

        x_train = (train_df[FEATURE_NAMES].to_numpy(dtype=np.float64) - feature_mean) / safe_std
        y_train = train_df["label"].map(label_to_index).to_numpy()
        x_test = (test_df[FEATURE_NAMES].to_numpy(dtype=np.float64) - feature_mean) / safe_std
        y_test = test_df["label"].map(label_to_index).to_numpy()

        raw_acc, bal_acc, matrix, y_pred = train_one_fold(
            x_train, y_train, x_test, y_test, args, fold_seed=args.seed + fold
        )
        bin_acc = binary_accuracy(y_test, y_pred, normal_index)

        raw_accs.append(raw_acc)
        bal_accs.append(bal_acc)
        bin_accs.append(bin_acc)
        total_matrix += matrix
        print(f"{fold:>5}{len(train_df):>9}{len(test_df):>8}{raw_acc:>10.4f}{bal_acc:>10.4f}{bin_acc:>10.4f}")

    raw_accs, bal_accs, bin_accs = map(np.array, (raw_accs, bal_accs, bin_accs))
    print(f"\n=== {args.n_folds}-fold (agrupado por ensaio), pipeline v2 ===")
    print(f"acuracia bruta:      {raw_accs.mean():.4f} +/- {raw_accs.std():.4f}")
    print(f"acuracia balanceada: {bal_accs.mean():.4f} +/- {bal_accs.std():.4f}")
    print(f"acuracia binaria:    {bin_accs.mean():.4f} +/- {bin_accs.std():.4f}")
    print("\nmatriz de confusao somada (todos os folds, cada fonte aparece 1x no teste):")
    print("            " + "  ".join(f"{n[:10]:>10s}" for n in CLASS_NAMES))
    for i, row in enumerate(total_matrix):
        print(f"{CLASS_NAMES[i][:10]:>10s}  " + "  ".join(f"{v:10d}" for v in row))

    per_class = per_class_precision_recall_f1(total_matrix, CLASS_NAMES)
    print("\nprecisao/recall/f1 por classe (sobre a matriz somada):")
    print(f"{'classe':>20}{'precisao':>10}{'recall':>10}{'f1':>10}{'suporte':>10}")
    for name, m in per_class.items():
        print(f"{name:>20}{m['precision']:>10.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['support']:>10d}")

    report = {
        "pipeline_version": "v2", "hidden_units": args.hidden, "epochs": args.epochs, "learning_rate": args.lr,
        "weighting": args.weighting,
        "n_folds": args.n_folds, "feature_names": FEATURE_NAMES, "class_names": CLASS_NAMES,
        "raw_accuracy_per_fold": raw_accs.tolist(), "balanced_accuracy_per_fold": bal_accs.tolist(),
        "binary_accuracy_per_fold": bin_accs.tolist(),
        "raw_accuracy_mean": float(raw_accs.mean()), "raw_accuracy_std": float(raw_accs.std()),
        "balanced_accuracy_mean": float(bal_accs.mean()), "balanced_accuracy_std": float(bal_accs.std()),
        "binary_accuracy_mean": float(bin_accs.mean()), "binary_accuracy_std": float(bin_accs.std()),
        "confusion_matrix_summed": total_matrix.tolist(),
        "per_class_precision_recall_f1": per_class,
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {REPORT_OUTPUT}")

    if args.save_final:
        print(f"\n=== treino final sobre TODAS as {len(df)} linhas (sem retencao), pipeline v2 ===")
        feature_mean = df[FEATURE_NAMES].to_numpy(dtype=np.float64).mean(axis=0)
        feature_std = df[FEATURE_NAMES].to_numpy(dtype=np.float64).std(axis=0)
        safe_std = np.where(feature_std == 0, 1.0, feature_std)
        x_all = (df[FEATURE_NAMES].to_numpy(dtype=np.float64) - feature_mean) / safe_std
        y_all = df["label"].map(label_to_index).to_numpy()
        sample_weights_all = class_weights_from_counts(y_all, n_classes, scheme=args.weighting)[y_all]

        weights = init_weights(n_hidden=args.hidden, seed=args.seed)
        for epoch in range(args.epochs):
            _, _, grads = compute_loss_and_grads(x_all, y_all, weights, sample_weights=sample_weights_all)
            weights = apply_gradient_step(weights, grads, learning_rate=args.lr)

        np.savez(
            FINAL_WEIGHTS_OUTPUT,
            w1=weights.w1, b1=weights.b1, w2=weights.w2, b2=weights.b2,
            feature_mean=feature_mean, feature_std=feature_std,
            class_names=np.array(CLASS_NAMES), feature_names=np.array(FEATURE_NAMES),
        )
        final_report = {
            "pipeline_version": "v2",
            "hidden_units": args.hidden, "epochs": args.epochs, "learning_rate": args.lr,
            "class_names": CLASS_NAMES, "feature_names": FEATURE_NAMES,
            "trained_on": "all 45 sources, no held-out data -- see evaluation_method below",
            "evaluation_method": f"{args.n_folds}-fold cross-validation grouped by (label, "
                                  "fault_detail) -- these are NOT a held-out test "
                                  "of this exact model (there is none, by design); they are the "
                                  "honest generalization estimate this training configuration "
                                  "produces, averaged over 3 independent train/test rotations "
                                  "using every source in this dataset.",
            "test_raw_accuracy_mean": float(raw_accs.mean()), "test_raw_accuracy_std": float(raw_accs.std()),
            "test_balanced_accuracy_mean": float(bal_accs.mean()), "test_balanced_accuracy_std": float(bal_accs.std()),
            "test_binary_accuracy_mean": float(bin_accs.mean()), "test_binary_accuracy_std": float(bin_accs.std()),
            "kfold_confusion_matrix_summed": total_matrix.tolist(),
            "kfold_per_class_precision_recall_f1": per_class,
        }
        FINAL_REPORT_OUTPUT.write_text(json.dumps(final_report, indent=2))
        print(f"Wrote official v2 weights to {FINAL_WEIGHTS_OUTPUT}")
        print(f"Wrote official v2 report to {FINAL_REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
