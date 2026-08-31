"""Pipeline v2 counterpart to tools/build_features_dataset.py -- identical
source/block iteration, but runs features.pipeline_v2.extract_features
(v1's vector plus 4 AR(4) Yule-Walker coefficients via matrix_inv)
instead of v1's alone. Writes a SEPARATE parquet
(features_dataset_v2.parquet) -- v1's official features_dataset.parquet
is untouched.

Usage (from 03.Reference):
    python -m tools.build_features_dataset_v2 [--max-blocks-per-source N]
"""

import argparse
from pathlib import Path

import pandas as pd

from dataset.paths import DATASET_DIR, EDA_OUTPUT_DIR, RAW_DATASET_DIR
from features.pipeline import BLOCK_SAMPLES, N_VIBRATION_CHANNELS
from features.pipeline_v2 import FEATURE_NAMES, extract_features

# Raw .mat (vibration) and .tdms (current/temperature) files live together
# in one flat directory -- see dataset/paths.py.
VIBRATION_DIR = RAW_DATASET_DIR
CURRENT_DIR = RAW_DATASET_DIR
CURRENT_CHANNEL_NAME = "corrente_fase_u"
# same window-count fix as build_features_dataset.py -- see that file's comment.
DEFAULT_MAX_BLOCKS_PER_SOURCE = 10_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-blocks-per-source", type=int, default=DEFAULT_MAX_BLOCKS_PER_SOURCE)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "features_dataset_v2.parquet")
    return parser.parse_args()


def pair_sources():
    vib_files = {p.stem.replace("Unbalalnce", "Unbalance"): p for p in VIBRATION_DIR.glob("*.mat")}
    cur_files = {p.stem: p for p in CURRENT_DIR.glob("*.tdms")}
    common = sorted(set(vib_files) & set(cur_files))
    missing = sorted((set(vib_files) | set(cur_files)) - set(common))
    if missing:
        raise ValueError(f"unpaired sources: {missing}")
    return [(stem, vib_files[stem], cur_files[stem]) for stem in common]


def evenly_spaced_starts(n_samples, n_blocks_wanted):
    n_available = (n_samples - 1000) // BLOCK_SAMPLES
    n_blocks = min(n_blocks_wanted, n_available)
    if n_blocks <= 0:
        return []
    stride = max(1, n_available // n_blocks)
    return [1000 + i * stride * BLOCK_SAMPLES for i in range(n_blocks)]


def main():
    args = parse_args()
    from build_dataset import load_mat_signal, load_tdms_channels, parse_source_metadata

    split_df = pd.read_csv(EDA_OUTPUT_DIR / "train_val_test_split.csv")
    split_lookup = {(row.fault_detail, row.load_nm): (row.label, row.split) for row in split_df.itertuples()}
    folds_df = pd.read_csv(EDA_OUTPUT_DIR / "folds.csv")
    fold_lookup = {(row.fault_detail, row.load_nm): row.fold for row in folds_df.itertuples()}

    pairs = pair_sources()
    print(f"Found {len(pairs)} paired sources (pipeline v2: v1 + AR(4) via matrix_inv)")

    rows = []
    for index, (stem, mat_path, tdms_path) in enumerate(pairs, start=1):
        load_nm = int(stem.split("Nm_")[0])
        meta = parse_source_metadata(stem)
        key = (meta.fault_detail, load_nm)
        if key not in split_lookup:
            print(f"[{index}/{len(pairs)}] {stem}: no split entry for {key}, skipping")
            continue
        label, split = split_lookup[key]
        fold = fold_lookup[key]

        vib_mat = load_mat_signal(mat_path).values  # (n, 4): x_A, y_A, x_B, y_B
        cur_channels = load_tdms_channels(tdms_path)
        cur_ch = next((c for c in cur_channels if c.column_name == CURRENT_CHANNEL_NAME), None)
        if cur_ch is None:
            print(f"[{index}/{len(pairs)}] {stem}: no {CURRENT_CHANNEL_NAME}, skipping")
            continue
        cur = cur_ch.values

        n_samples = min(len(vib_mat), len(cur))
        starts = evenly_spaced_starts(n_samples, args.max_blocks_per_source)
        print(f"[{index}/{len(pairs)}] {stem}: {len(starts)} blocks")

        for start in starts:
            vib_blocks = [vib_mat[start : start + BLOCK_SAMPLES, c] for c in range(N_VIBRATION_CHANNELS)]
            cur_block = cur[start : start + BLOCK_SAMPLES]
            features = extract_features(vib_blocks, cur_block)
            rows.append(
                {
                    "label": label,
                    "fault_detail": meta.fault_detail,
                    "load_nm": load_nm,
                    "split": split,
                    "fold": fold,
                    **dict(zip(FEATURE_NAMES, features.as_tuple())),
                }
            )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print()
    print(df.groupby(["label", "fold"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
