"""Builds a spectrogram dataset from cnn.reference.build_lowfreq_spectrogram
(decimated) instead of the native-resolution one (cnn.reference.
build_spectrogram), so the CNN can see the rotational harmonics
(1x/2x/3x, ~50/100/150Hz) the native spectrogram buries in the
discarded DC bin.

Defaults to 2 channels (x_mancal_a, y_mancal_a -- "mancal A", the
biaxial accelerometer on one bearing housing): the adopted official
config after comparing 1/2/4 channels and a two-model ensemble --
2 channels wins on every metric with no trade-off, while 4 channels
(and an ensemble of a mancal-A model with a mancal-B model) both
destabilize the rarest class (operacao_normal) through overfitting,
the same mechanism that made a single native-resolution channel win
over 4 in the very first channel-count comparison. --channels lets you
build other subsets to double-check.

Each spectrogram needs LOWFREQ_BLOCK_SAMPLES (~1.4s) of raw signal
instead of ~80ms for the native-resolution version -- far fewer
non-overlapping spectrograms fit in a given source, so this defaults to
fewer blocks/source than the native spectrogram builder.

Usage (from 03.Reference):
    python -m tools.build_spectrogram_dataset_lowfreq [--channels 0 1] [--max-blocks-per-source N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cnn.reference import LOWFREQ_BLOCK_SAMPLES, build_lowfreq_spectrogram_multichannel
from dataset.paths import DATASET_DIR, EDA_OUTPUT_DIR, RAW_DATASET_DIR
from dataset.signal_params import (
    VIBRATION_INGESTION_SCALE,
    VIBRATION_XB_INGESTION_SCALE,
    VIBRATION_Y_INGESTION_SCALE,
    VIBRATION_YB_INGESTION_SCALE,
    normalize_ingestion,
)

# Raw .mat (vibration) files live in one flat directory -- see dataset/paths.py.
VIBRATION_DIR = RAW_DATASET_DIR
# column order in the raw .mat: x_A, y_A, x_B, y_B (build_dataset.py)
CHANNEL_SCALES = [
    VIBRATION_INGESTION_SCALE,
    VIBRATION_Y_INGESTION_SCALE,
    VIBRATION_XB_INGESTION_SCALE,
    VIBRATION_YB_INGESTION_SCALE,
]
# A fixed per-source cap smaller than a source's real duration silently
# discards recorded signal instead of using all of it -- see
# build_features_dataset.py's window-count comment for the measurement
# that motivated raising this well above any observed per-source ceiling
# (so evenly_spaced_starts's min(wanted, available) always resolves to
# "available").
DEFAULT_MAX_BLOCKS_PER_SOURCE = 10_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blocks-per-source", type=int, default=DEFAULT_MAX_BLOCKS_PER_SOURCE)
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1],
                         help="vibration channel indices (0=x_A, 1=y_A, 2=x_B, 3=y_B). Default: [0, 1] (mancal A)")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evenly_spaced_starts(n_samples, n_blocks_wanted):
    n_available = (n_samples - 1000) // LOWFREQ_BLOCK_SAMPLES
    n_blocks = min(n_blocks_wanted, n_available)
    if n_blocks <= 0:
        return []
    stride = max(1, n_available // n_blocks)
    return [1000 + i * stride * LOWFREQ_BLOCK_SAMPLES for i in range(n_blocks)]


def main():
    args = parse_args()
    from build_dataset import load_mat_signal, parse_source_metadata

    suffix = "-".join(str(c) for c in args.channels)
    output = args.output or DATASET_DIR / f"spectrogram_dataset_lowfreq_ch{suffix}.npz"

    split_df = pd.read_csv(EDA_OUTPUT_DIR / "train_val_test_split.csv")
    split_lookup = {(row.fault_detail, row.load_nm): (row.label, row.split) for row in split_df.itertuples()}
    folds_df = pd.read_csv(EDA_OUTPUT_DIR / "folds.csv")
    fold_lookup = {(row.fault_detail, row.load_nm): row.fold for row in folds_df.itertuples()}

    vib_files = {p.stem.replace("Unbalalnce", "Unbalance"): p for p in VIBRATION_DIR.glob("*.mat")}
    stems = sorted(vib_files)
    print(f"Found {len(stems)} vibration sources; channels: {args.channels}; "
          f"LOWFREQ_BLOCK_SAMPLES={LOWFREQ_BLOCK_SAMPLES} ({LOWFREQ_BLOCK_SAMPLES/25600*1000:.0f}ms)")

    spectrograms, labels, fault_details, load_nms, splits, folds = [], [], [], [], [], []
    for index, stem in enumerate(stems, start=1):
        load_nm = int(stem.split("Nm_")[0])
        meta = parse_source_metadata(stem)
        key = (meta.fault_detail, load_nm)
        if key not in split_lookup:
            print(f"[{index}/{len(stems)}] {stem}: no split entry, skipping")
            continue
        label, split = split_lookup[key]
        fold = fold_lookup[key]

        vib_mat = load_mat_signal(vib_files[stem]).values  # (n, 4): x_A, y_A, x_B, y_B
        starts = evenly_spaced_starts(len(vib_mat), args.max_blocks_per_source)
        print(f"[{index}/{len(stems)}] {stem}: {len(starts)} spectrograms")

        for start in starts:
            blocks = [
                normalize_ingestion(vib_mat[start : start + LOWFREQ_BLOCK_SAMPLES, ch], CHANNEL_SCALES[ch])
                for ch in args.channels
            ]
            spectrograms.append(build_lowfreq_spectrogram_multichannel(blocks))  # (C, 32, 32)
            labels.append(label)
            fault_details.append(meta.fault_detail)
            load_nms.append(load_nm)
            splits.append(split)
            folds.append(fold)

    x = np.stack(spectrograms).astype(np.float32)  # (N, C, 32, 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        x=x,
        label=np.array(labels),
        fault_detail=np.array(fault_details),
        load_nm=np.array(load_nms),
        split=np.array(splits),
        fold=np.array(folds),
    )
    print(f"\nWrote {len(labels)} spectrograms ({x.shape}) to {output}")

    df = pd.DataFrame({"label": labels, "split": splits, "fold": folds})
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print()
    print(df.groupby(["label", "fold"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
