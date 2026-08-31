"""Generates eda/folds.csv from dataset.split.assign_folds -- grouped
k-fold cross-validation, sibling of tools/build_train_val_test_split.py
(the fixed-split version, kept as a simpler baseline).

Usage (from 03.Reference):
    python -m tools.build_folds [--n-folds 3]
"""

import argparse

from dataset.loader import open_connection
from dataset.paths import EDA_OUTPUT_DIR
from dataset.split import assign_folds

OUTPUT_CSV = EDA_OUTPUT_DIR / "folds.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    con = open_connection()
    result = assign_folds(con, seed=args.seed, n_folds=args.n_folds)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(result)} source rows to {OUTPUT_CSV}")
    print(result.groupby(["label", "fold"]).size().unstack(fill_value=0))
    print()
    print("Per-severidade (cada grupo deve ter 1 fonte por fold):")
    print(result.groupby(["label", "fault_detail", "fold"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
