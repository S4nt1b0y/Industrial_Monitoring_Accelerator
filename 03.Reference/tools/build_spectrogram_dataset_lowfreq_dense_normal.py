"""CNN counterpart to tools/build_features_dataset_dense_normal.py --
overlapping spectrogram windows for operacao_normal's 3 sources only,
on top of the non-overlapping ceiling every other class already uses.
Everything else is copied unchanged from the base (non-dense)
spectrogram dataset. Same rationale as the MLP version (a stationary
process -> denser sampling of the SAME held-in sources sharpens the
model's estimate of what "normal" looks like; no leakage, since grouped
k-fold still keeps a whole source in one fold), and it produces the
same kind of gain here (operacao_normal precision roughly doubles) on
the CNN's adopted config (mancal A, channels 0,1).

Usage (from 03.Reference):
    python -m tools.build_spectrogram_dataset_lowfreq_dense_normal --overlap-divisor 4 --channels 0 1
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
CHANNEL_SCALES = [VIBRATION_INGESTION_SCALE, VIBRATION_Y_INGESTION_SCALE,
                  VIBRATION_XB_INGESTION_SCALE, VIBRATION_YB_INGESTION_SCALE]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-divisor", type=int, default=4)
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--base-spectrogram", type=Path, default=None,
                         help="default: spectrogram_dataset_lowfreq_ch<channels>.npz")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def dense_starts(n_samples, hop):
    n_available = (n_samples - 1000 - LOWFREQ_BLOCK_SAMPLES) // hop + 1
    return [1000 + i * hop for i in range(max(n_available, 0))]


def main():
    args = parse_args()
    from build_dataset import load_mat_signal, parse_source_metadata

    suffix = "-".join(str(c) for c in args.channels)
    base_spectrogram = args.base_spectrogram or DATASET_DIR / f"spectrogram_dataset_lowfreq_ch{suffix}.npz"
    output = args.output or DATASET_DIR / f"spectrogram_dataset_lowfreq_ch{suffix}_dense_normal.npz"
    hop = LOWFREQ_BLOCK_SAMPLES // args.overlap_divisor
    print(f"channels={args.channels} LOWFREQ_BLOCK_SAMPLES={LOWFREQ_BLOCK_SAMPLES} hop={hop} (divisor={args.overlap_divisor})")

    split_df = pd.read_csv(EDA_OUTPUT_DIR / "train_val_test_split.csv")
    split_lookup = {(row.fault_detail, row.load_nm): (row.label, row.split) for row in split_df.itertuples()}
    folds_df = pd.read_csv(EDA_OUTPUT_DIR / "folds.csv")
    fold_lookup = {(row.fault_detail, row.load_nm): row.fold for row in folds_df.itertuples()}

    vib_files = {p.stem.replace("Unbalalnce", "Unbalance"): p for p in VIBRATION_DIR.glob("*.mat")}

    spectrograms, labels, fault_details, load_nms, splits, folds = [], [], [], [], [], []
    for stem, path in sorted(vib_files.items()):
        load_nm = int(stem.split("Nm_")[0])
        meta = parse_source_metadata(stem)
        if meta.label != "operacao_normal":
            continue
        key = (meta.fault_detail, load_nm)
        if key not in split_lookup:
            continue
        label, split = split_lookup[key]
        fold = fold_lookup[key]

        vib_mat = load_mat_signal(path).values
        starts = dense_starts(len(vib_mat), hop)
        print(f"{stem}: {len(starts)} overlapping spectrograms (hop={hop})")

        for start in starts:
            blocks = [
                normalize_ingestion(vib_mat[start : start + LOWFREQ_BLOCK_SAMPLES, ch], CHANNEL_SCALES[ch])
                for ch in args.channels
            ]
            spectrograms.append(build_lowfreq_spectrogram_multichannel(blocks))
            labels.append(label)
            fault_details.append(meta.fault_detail)
            load_nms.append(load_nm)
            splits.append(split)
            folds.append(fold)

    dense_x = np.stack(spectrograms).astype(np.float32)
    dense_label = np.array(labels)

    base = np.load(base_spectrogram, allow_pickle=True)
    keep_mask = base["label"] != "operacao_normal"
    combined_x = np.concatenate([base["x"][keep_mask], dense_x], axis=0)
    combined_label = np.concatenate([base["label"][keep_mask], dense_label])
    combined_fault_detail = np.concatenate([base["fault_detail"][keep_mask], np.array(fault_details)])
    combined_load_nm = np.concatenate([base["load_nm"][keep_mask], np.array(load_nms)])
    combined_split = np.concatenate([base["split"][keep_mask], np.array(splits)])
    combined_fold = np.concatenate([base["fold"][keep_mask], np.array(folds)])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x=combined_x, label=combined_label, fault_detail=combined_fault_detail,
              load_nm=combined_load_nm, split=combined_split, fold=combined_fold)
    n_before = int((base["label"] == "operacao_normal").sum())
    print(f"\noperacao_normal: {n_before} -> {len(dense_label)} spectrograms")
    print(f"Wrote {len(combined_label)} spectrograms ({combined_x.shape}) to {output}")


if __name__ == "__main__":
    main()
