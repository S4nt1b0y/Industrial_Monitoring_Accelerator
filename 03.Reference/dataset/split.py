"""Train/val/test split at the *source* level, grouped by (label,
fault_detail).

Splitting by row would leak information: rows from the same recording are
temporally correlated (adjacent samples of the same 60-300s run), so a
row-level random split would put near-duplicate windows in both train and
test. Every row from a given (fault_detail, load_nm) source goes entirely
into one split.

Grouping by (label, fault_detail) rather than by label alone means every
severity within a class gets its own train/val/test allocation, instead
of a class's severities landing unevenly across splits by chance (a
class's weakest, least separable severity could otherwise end up
entirely in test, making the reported metric harder than the class
actually is). With exactly 3 loads (0/2/4 Nm) per (label, fault_detail)
in this dataset, `_allocate_indices`'s "guarantee >=1 per split when
n>=3" floor gives every severity a 1/1/1 split.
"""

import numpy as np

from .loader import TABLE_NAME

SPLIT_NAMES = ("train", "val", "test")


def _allocate_indices(n, val_frac, test_frac, rng):
    order = rng.permutation(n)
    if n >= 3:
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_val = min(n_val, n - n_test - 1) if n - n_test - 1 >= 1 else n_val
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = n - 2, 1, 1
    else:
        n_train, n_val, n_test = n, 0, 0

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx


def assign_splits(con, table=TABLE_NAME, seed=0, val_frac=0.15, test_frac=0.15):
    """Return a DataFrame with one row per source and its assigned split."""
    sources = con.execute(f"""
        select label, fault_detail, load_nm, count(*) as rows
        from {table}
        group by label, fault_detail, load_nm
        order by label, fault_detail, load_nm
    """).df()

    rng = np.random.default_rng(seed)
    split_assignment = np.empty(len(sources), dtype=object)

    for (label, fault_detail), group in sources.groupby(["label", "fault_detail"], sort=False):
        idx = group.index.to_numpy()
        train_idx, val_idx, test_idx = _allocate_indices(len(idx), val_frac, test_frac, rng)
        split_assignment[idx[train_idx]] = "train"
        split_assignment[idx[val_idx]] = "val"
        split_assignment[idx[test_idx]] = "test"

    sources["split"] = split_assignment
    return sources


def assign_folds(con, table=TABLE_NAME, seed=0, n_folds=3):
    """Return a DataFrame with one row per source and a `fold` column
    (0 .. n_folds-1) -- grouped k-fold cross-validation, grouped by
    (label, fault_detail) for the same reason as assign_splits: every
    severity gets representation in every fold, not just by luck.

    Unlike assign_splits, no source is permanently reserved for val/test --
    a fold i's "test" set is that fold's sources; everything else trains.
    Across all n_folds rotations every source has been in the training set
    n_folds-1 times and the test set once, using the full dataset far
    more efficiently than a fixed train/val/test split, which would
    permanently waste most sources on val+test.

    With exactly n_folds sources per (label, fault_detail) group (this
    dataset: always 3 loads), each group lands exactly 1 source per fold.
    Groups with fewer sources than n_folds distribute round-robin over a
    shuffled order instead (no guarantee every fold sees them, but no
    dataset table this project uses has that case today).
    """
    sources = con.execute(f"""
        select label, fault_detail, load_nm, count(*) as rows
        from {table}
        group by label, fault_detail, load_nm
        order by label, fault_detail, load_nm
    """).df()

    rng = np.random.default_rng(seed)
    fold_assignment = np.empty(len(sources), dtype=int)

    for (label, fault_detail), group in sources.groupby(["label", "fault_detail"], sort=False):
        idx = group.index.to_numpy()
        order = rng.permutation(len(idx))
        fold_assignment[idx[order]] = np.arange(len(idx)) % n_folds

    sources["fold"] = fold_assignment
    return sources
