"""(Re)generates eda/train_val_test_split.csv from
dataset.split.assign_splits, grouped by (label, fault_detail) -- see
dataset/split.py's docstring for why grouping matters.

Usage (from 03.Reference):
    python -m tools.build_train_val_test_split
"""

from dataset.loader import open_connection
from dataset.paths import EDA_OUTPUT_DIR
from dataset.split import assign_splits

OUTPUT_CSV = EDA_OUTPUT_DIR / "train_val_test_split.csv"


def main():
    con = open_connection()
    result = assign_splits(con, seed=0)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(result)} source rows to {OUTPUT_CSV}")
    print(result.groupby(["label", "split"]).size().unstack(fill_value=0))
    print()
    print("Per-severity split (every severity should appear in every split):")
    print(result.groupby(["label", "fault_detail", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
