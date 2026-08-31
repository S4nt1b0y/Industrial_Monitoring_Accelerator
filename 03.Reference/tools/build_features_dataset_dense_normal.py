"""Overlapping windows for operacao_normal's 3 sources only, on top of
the non-overlapping ceiling every other class already uses (see
build_features_dataset.py's window-count comment). Every other class is
left untouched (copied straight from the base parquet) -- operacao_normal
is the one class scarce enough (3 recordings total) that a denser,
overlapping sampling of the SAME recordings measurably sharpens the
model's estimate of what that class looks like, without touching the
other 3 classes' data at all.

Overlap divisor D means hop = BLOCK_SAMPLES // D (D=1 reproduces the
non-overlapping set exactly, as a built-in sanity check). No leakage
risk: grouped k-fold assigns a whole SOURCE to one fold -- every extra
overlapping window from a given source still belongs to that same
source's fold, same as the non-overlapping ones.

Usage (from 03.Reference):
    python -m tools.build_features_dataset_dense_normal --overlap-divisor 4 --pipeline v2
"""

import argparse
from pathlib import Path

import pandas as pd

from dataset.paths import DATASET_DIR, EDA_OUTPUT_DIR, RAW_DATASET_DIR
from features.pipeline import BLOCK_SAMPLES, N_VIBRATION_CHANNELS

# Raw .mat (vibration) and .tdms (current/temperature) files live together
# in one flat directory -- see dataset/paths.py.
VIBRATION_DIR = RAW_DATASET_DIR
CURRENT_DIR = RAW_DATASET_DIR
CURRENT_CHANNEL_NAME = "corrente_fase_u"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-divisor", type=int, default=4,
                         help="hop = BLOCK_SAMPLES // divisor; 4 means 75% overlap between neighbors")
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v2")
    parser.add_argument("--labels", type=str, nargs="+", default=["operacao_normal"],
                         help="which classes get the dense/overlapping treatment; "
                              "everything else is copied unchanged from the base parquet")
    return parser.parse_args()


def dense_starts(n_samples, hop):
    n_available = (n_samples - 1000 - BLOCK_SAMPLES) // hop + 1
    return [1000 + i * hop for i in range(max(n_available, 0))]


def pair_sources():
    vib_files = {p.stem.replace("Unbalalnce", "Unbalance"): p for p in VIBRATION_DIR.glob("*.mat")}
    cur_files = {p.stem: p for p in CURRENT_DIR.glob("*.tdms")}
    common = sorted(set(vib_files) & set(cur_files))
    return [(stem, vib_files[stem], cur_files[stem]) for stem in common]


def main():
    args = parse_args()
    from build_dataset import load_mat_signal, load_tdms_channels, parse_source_metadata

    if args.pipeline == "v1":
        from features.pipeline import FEATURE_NAMES, extract_features
        base_parquet = DATASET_DIR / "features_dataset.parquet"
        output = DATASET_DIR / "features_dataset_dense_normal.parquet"
    else:
        from features.pipeline_v2 import FEATURE_NAMES, extract_features
        base_parquet = DATASET_DIR / "features_dataset_v2.parquet"
        output = DATASET_DIR / "features_dataset_v2_dense_normal.parquet"

    hop = BLOCK_SAMPLES // args.overlap_divisor
    print(f"pipeline={args.pipeline} BLOCK_SAMPLES={BLOCK_SAMPLES} hop={hop} (divisor={args.overlap_divisor})")

    split_df = pd.read_csv(EDA_OUTPUT_DIR / "train_val_test_split.csv")
    split_lookup = {(row.fault_detail, row.load_nm): (row.label, row.split) for row in split_df.itertuples()}
    folds_df = pd.read_csv(EDA_OUTPUT_DIR / "folds.csv")
    fold_lookup = {(row.fault_detail, row.load_nm): row.fold for row in folds_df.itertuples()}

    rows = []
    for stem, mat_path, tdms_path in pair_sources():
        load_nm = int(stem.split("Nm_")[0])
        meta = parse_source_metadata(stem)
        if meta.label not in args.labels:
            continue
        key = (meta.fault_detail, load_nm)
        if key not in split_lookup:
            continue
        label, split = split_lookup[key]
        fold = fold_lookup[key]

        vib_mat = load_mat_signal(mat_path).values
        cur_channels = load_tdms_channels(tdms_path)
        cur_ch = next((c for c in cur_channels if c.column_name == CURRENT_CHANNEL_NAME), None)
        cur = cur_ch.values
        n_samples = min(len(vib_mat), len(cur))

        starts = dense_starts(n_samples, hop)
        print(f"{stem}: {len(starts)} overlapping windows (hop={hop})")
        for start in starts:
            vib_blocks = [vib_mat[start : start + BLOCK_SAMPLES, c] for c in range(N_VIBRATION_CHANNELS)]
            cur_block = cur[start : start + BLOCK_SAMPLES]
            features = extract_features(vib_blocks, cur_block)
            rows.append({
                "label": label, "fault_detail": meta.fault_detail, "load_nm": load_nm,
                "split": split, "fold": fold,
                **dict(zip(FEATURE_NAMES, features.as_tuple())),
            })

    dense_df = pd.DataFrame(rows)
    base_df = pd.read_parquet(base_parquet)
    other_classes_df = base_df[~base_df["label"].isin(args.labels)]
    combined = pd.concat([other_classes_df, dense_df], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    for label in args.labels:
        before = len(base_df[base_df["label"] == label])
        after = len(dense_df[dense_df["label"] == label])
        print(f"{label}: {before} -> {after} windows")
    print(f"Wrote {len(combined)} rows to {output}")
    print(combined.groupby(["label", "fold"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
