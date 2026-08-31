"""Trains cnn.reference's CNN with grouped k-fold cross-validation
(dataset.split.assign_folds, tools/build_folds.py) -- every recording
rotates through the test role exactly once, using the full dataset far
more efficiently than a fixed train/val/test split.

Uses cnn.reference.compute_loss_and_grads_batched, not
compute_loss_and_grads: the per-sample version is far too slow for a
full k-fold comparison at this dataset's size; the batched version
(im2col, cross-validated to machine precision against the per-sample
version in tests/test_cnn.py) does the same math much faster.

Supports both spectrogram constructions (cnn.reference.build_spectrogram
/ build_lowfreq_spectrogram) via --spectrogram/--output-prefix -- the
low-frequency one is the default and the adopted configuration.

The spectrogram input is always z-score normalized (against the train
fold), same convention as every MLP training script. This matters more
than it looks: a fixed-scale weight initialization only works by
coincidence at one particular input scale, and training on a
spectrogram with a different natural scale collapses to predicting only
the majority class without this normalization.

Usage (from 03.Reference):
    python -m tools.train_cnn_kfold [--epochs 150] [--lr 0.01] [--save-final]
    python -m tools.train_cnn_kfold --spectrogram <path> --output-prefix cnn_alt
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cnn.reference import apply_gradient_step, compute_loss_and_grads_batched, init_weights, predict_batched
from dataset.paths import DATASET_DIR
from ml_classifier.reference import CLASS_NAMES
from tools.train_cnn import (
    balanced_accuracy,
    binary_accuracy,
    class_weights_from_counts,
    confusion_matrix,
    per_class_precision_recall_f1,
)

SPECTROGRAM_NPZ = DATASET_DIR / "spectrogram_dataset_lowfreq_ch0-1.npz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--save-final", action="store_true",
                         help="after the k-fold comparison, train one more model on ALL rows "
                              "and save it as the official cnn_weights.npz/training_report.json "
                              "-- same reasoning as train_ml_classifier_kfold.py --save-final")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                         help="subset of the spectrogram's channels to use (0=x_A, 1=y_A, "
                              "2=x_B, 3=y_B), e.g. --channels 0 1 for x_A+y_A only. Default: "
                              "every channel present in the chosen --spectrogram file.")
    parser.add_argument("--spectrogram", type=Path, default=SPECTROGRAM_NPZ,
                         help="path to the spectrogram .npz")
    parser.add_argument("--output-prefix", type=str, default="cnn",
                         help="filename prefix for report/weights outputs (default 'cnn' -- "
                              "the official filenames; use another prefix to write separate "
                              "files without touching the official ones)")
    parser.add_argument("--weighting", type=str, default="inverse",
                         choices=["inverse", "sqrt", "none", "capped"],
                         help="class-weighting scheme, see "
                              "tools/train_cnn.class_weights_from_counts")
    return parser.parse_args()


def normalize_spectrogram(x, mean, std):
    return (x - mean) / std


def train_one_fold(x_train, y_train, x_test, y_test, args, fold_seed, n_classes, n_channels):
    x_mean, x_std = x_train.mean(), x_train.std()
    safe_std = x_std if x_std > 0 else 1.0
    x_train = normalize_spectrogram(x_train, x_mean, safe_std)
    x_test = normalize_spectrogram(x_test, x_mean, safe_std)

    sample_weights_train = class_weights_from_counts(y_train, n_classes, scheme=args.weighting)[y_train]
    weights = init_weights(n_channels=n_channels, seed=fold_seed)

    for epoch in range(args.epochs):
        _, _, grads = compute_loss_and_grads_batched(x_train, y_train, weights, sample_weights=sample_weights_train)
        weights = apply_gradient_step(weights, grads, learning_rate=args.lr)

    y_pred = predict_batched(x_test, weights)
    matrix = confusion_matrix(y_test, y_pred, n_classes)
    raw_acc = float(np.mean(y_pred == y_test))
    bal_acc = balanced_accuracy(y_test, y_pred, n_classes)
    return raw_acc, bal_acc, matrix, y_pred


def main():
    args = parse_args()
    report_output = DATASET_DIR / f"{args.output_prefix}_kfold_report.json"
    final_weights_output = DATASET_DIR / f"{args.output_prefix}_weights.npz"
    final_report_output = DATASET_DIR / f"{args.output_prefix}_training_report.json"

    label_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    normal_index = label_to_index["operacao_normal"]
    n_classes = len(CLASS_NAMES)

    data = np.load(args.spectrogram, allow_pickle=True)
    x_all, label_all, fold_all = data["x"], data["label"], data["fold"]
    y_all = np.array([label_to_index[l] for l in label_all])

    channels = args.channels if args.channels is not None else list(range(x_all.shape[1]))
    n_channels = len(channels)
    x_all = x_all[:, channels, :, :]
    print(f"canais usados: {channels} ({n_channels} de {data['x'].shape[1]})")

    raw_accs, bal_accs, bin_accs = [], [], []
    total_matrix = np.zeros((n_classes, n_classes), dtype=int)

    print(f"{'fold':>5}{'n_train':>9}{'n_test':>8}{'raw_acc':>10}{'bal_acc':>10}{'bin_acc':>10}")
    for fold in range(args.n_folds):
        test_mask = fold_all == fold
        train_mask = ~test_mask
        x_train, y_train = x_all[train_mask], y_all[train_mask]
        x_test, y_test = x_all[test_mask], y_all[test_mask]

        raw_acc, bal_acc, matrix, y_pred = train_one_fold(
            x_train, y_train, x_test, y_test, args, fold_seed=args.seed + fold,
            n_classes=n_classes, n_channels=n_channels
        )
        bin_acc = binary_accuracy(y_test, y_pred, normal_index)

        raw_accs.append(raw_acc)
        bal_accs.append(bal_acc)
        bin_accs.append(bin_acc)
        total_matrix += matrix
        print(f"{fold:>5}{len(y_train):>9}{len(y_test):>8}{raw_acc:>10.4f}{bal_acc:>10.4f}{bin_acc:>10.4f}")

    raw_accs, bal_accs, bin_accs = map(np.array, (raw_accs, bal_accs, bin_accs))
    print(f"\n=== {args.n_folds}-fold (agrupado por ensaio) ===")
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
        "epochs": args.epochs, "learning_rate": args.lr, "n_folds": args.n_folds,
        "channels": channels, "class_names": CLASS_NAMES,
        "raw_accuracy_per_fold": raw_accs.tolist(), "balanced_accuracy_per_fold": bal_accs.tolist(),
        "binary_accuracy_per_fold": bin_accs.tolist(),
        "raw_accuracy_mean": float(raw_accs.mean()), "raw_accuracy_std": float(raw_accs.std()),
        "balanced_accuracy_mean": float(bal_accs.mean()), "balanced_accuracy_std": float(bal_accs.std()),
        "binary_accuracy_mean": float(bin_accs.mean()), "binary_accuracy_std": float(bin_accs.std()),
        "confusion_matrix_summed": total_matrix.tolist(),
        "per_class_precision_recall_f1": per_class,
    }
    report_output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {report_output}")

    if args.save_final:
        print(f"\n=== treino final sobre TODAS as {len(y_all)} linhas (sem retencao) ===")
        x_mean, x_std = x_all.mean(), x_all.std()
        safe_std = x_std if x_std > 0 else 1.0
        x_all_norm = normalize_spectrogram(x_all, x_mean, safe_std)

        sample_weights_all = class_weights_from_counts(y_all, n_classes, scheme=args.weighting)[y_all]
        weights = init_weights(n_channels=n_channels, seed=args.seed)
        for epoch in range(args.epochs):
            _, _, grads = compute_loss_and_grads_batched(x_all_norm, y_all, weights, sample_weights=sample_weights_all)
            weights = apply_gradient_step(weights, grads, learning_rate=args.lr)

        np.savez(final_weights_output, conv1_kernels=weights.conv1_kernels, conv1_bias=weights.conv1_bias,
                 dense_w=weights.dense_w, dense_b=weights.dense_b, class_names=np.array(CLASS_NAMES),
                 spectrogram_mean=np.float64(x_mean), spectrogram_std=np.float64(safe_std))
        final_report = {
            "epochs": args.epochs, "learning_rate": args.lr, "class_names": CLASS_NAMES,
            "trained_on": "all sources, no held-out data -- see evaluation_method below",
            "evaluation_method": f"{args.n_folds}-fold cross-validation grouped by (label, "
                                  "fault_detail) -- NOT a held-out test of this "
                                  "exact model (there is none, by design); the honest "
                                  "generalization estimate for this training configuration.",
            "test_raw_accuracy_mean": float(raw_accs.mean()), "test_raw_accuracy_std": float(raw_accs.std()),
            "test_balanced_accuracy_mean": float(bal_accs.mean()), "test_balanced_accuracy_std": float(bal_accs.std()),
            "test_binary_accuracy_mean": float(bin_accs.mean()), "test_binary_accuracy_std": float(bin_accs.std()),
            "kfold_confusion_matrix_summed": total_matrix.tolist(),
            "kfold_per_class_precision_recall_f1": per_class,
        }
        final_report_output.write_text(json.dumps(final_report, indent=2))
        print(f"Wrote official weights to {final_weights_output}")
        print(f"Wrote official report to {final_report_output}")


if __name__ == "__main__":
    main()
