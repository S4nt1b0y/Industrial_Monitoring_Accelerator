import duckdb
import pandas as pd
import pytest

from dataset.loader import TABLE_NAME
from dataset.split import SPLIT_NAMES, assign_folds, assign_splits


def _make_sources_table(severities_per_label):
    """severities_per_label: {label: {fault_detail: n_sources}} -> a table
    with 1 row per source, mirroring the real dataset's shape (a
    fault_detail/severity shared by several sources, one per load_nm)."""
    rows = []
    for label, severities in severities_per_label.items():
        for fault_detail, n_sources in severities.items():
            for i in range(n_sources):
                rows.append((label, fault_detail, i, 100))
    return pd.DataFrame(rows, columns=["label", "fault_detail", "load_nm", "rows"])


@pytest.fixture
def con_factory():
    def make(rows_per_label):
        sources = _make_sources_table(rows_per_label)
        expanded = sources.loc[sources.index.repeat(sources["rows"])].drop(columns=["rows"])
        con = duckdb.connect()
        con.register("fixture_df", expanded)
        con.execute(f"create table {TABLE_NAME} as select * from fixture_df")
        return con

    return make


def test_every_source_gets_exactly_one_split(con_factory):
    con = con_factory({
        "operacao_normal": {"normal": 3},
        "desgaste_rolamento": {"BPFO_03": 3, "BPFO_10": 3, "BPFO_30": 3,
                                "BPFI_03": 3, "BPFI_10": 3, "BPFI_30": 3},
    })
    result = assign_splits(con, seed=1)
    assert set(result["split"]) <= set(SPLIT_NAMES)
    assert result["split"].isna().sum() == 0


def test_small_class_still_gets_all_three_splits(con_factory):
    con = con_factory({"operacao_normal": {"normal": 3}})
    result = assign_splits(con, seed=1)
    assert set(result["split"]) == {"train", "val", "test"}


def test_split_is_deterministic_for_a_fixed_seed(con_factory):
    con = con_factory({
        "desbalanceamento": {f"unbalance_{i}": 3 for i in range(5)},
        "desalinhamento": {f"misalign_{i}": 3 for i in range(3)},
    })
    first = assign_splits(con, seed=42)
    second = assign_splits(con, seed=42)
    pd.testing.assert_series_equal(first["split"], second["split"])


def test_every_severity_appears_in_every_split(con_factory):
    # Regression test: stratifying by label alone let one severity
    # dominate val/test while train saw a different mix (real case:
    # desalinhamento's weakest severity was 100% of both val and test).
    # With 3 sources per
    # fault_detail, every severity should now land 1/1/1 across splits.
    con = con_factory({
        "desalinhamento": {"misalign_01": 3, "misalign_03": 3, "misalign_05": 3},
    })
    result = assign_splits(con, seed=1)
    for fault_detail, group in result.groupby("fault_detail"):
        assert set(group["split"]) == {"train", "val", "test"}, (
            f"{fault_detail} missing from some split: {sorted(group['split'])}"
        )


def test_assign_folds_every_source_gets_exactly_one_fold(con_factory):
    con = con_factory({
        "operacao_normal": {"normal": 3},
        "desgaste_rolamento": {"BPFO_03": 3, "BPFI_30": 3},
    })
    result = assign_folds(con, seed=1, n_folds=3)
    assert set(result["fold"]) <= {0, 1, 2}
    assert result["fold"].isna().sum() == 0


def test_assign_folds_every_severity_appears_in_every_fold(con_factory):
    # Same regression concern as the fixed split, applied to k-folds:
    # each fault_detail group has 3 sources and n_folds=3, so
    # every severity should land exactly one source per fold.
    con = con_factory({
        "desbalanceamento": {f"unbalance_{i}": 3 for i in range(5)},
        "desalinhamento": {f"misalign_{i}": 3 for i in range(3)},
    })
    result = assign_folds(con, seed=1, n_folds=3)
    for fault_detail, group in result.groupby("fault_detail"):
        assert set(group["fold"]) == {0, 1, 2}, (
            f"{fault_detail} missing from some fold: {sorted(group['fold'])}"
        )
        assert len(group) == 3


def test_assign_folds_uses_every_source_across_folds(con_factory):
    # The whole point of k-fold over a fixed split: no source is
    # permanently excluded from training -- each is in the test fold
    # exactly once and the training set for every other fold.
    con = con_factory({"desalinhamento": {"misalign_01": 3, "misalign_03": 3}})
    result = assign_folds(con, seed=1, n_folds=3)
    assert set(result["fold"].value_counts().index) == {0, 1, 2}
    assert (result["fold"].value_counts() == 2).all()  # 2 groups x 3 sources / 3 folds


def test_assign_folds_is_deterministic_for_a_fixed_seed(con_factory):
    con = con_factory({"desbalanceamento": {f"unbalance_{i}": 3 for i in range(5)}})
    first = assign_folds(con, seed=7)
    second = assign_folds(con, seed=7)
    pd.testing.assert_series_equal(first["fold"], second["fold"])
