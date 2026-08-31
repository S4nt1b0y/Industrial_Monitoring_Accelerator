import duckdb
import pandas as pd
import pytest

from dataset import eda
from dataset.loader import MEASUREMENT_COLUMNS, TABLE_NAME, VALIDITY_COLUMNS
from dataset.normalize import fit_zscore_params


@pytest.fixture
def synthetic_con():
    columns = ["label", "fault_detail", "load_nm"] + MEASUREMENT_COLUMNS + VALIDITY_COLUMNS
    rows = [
        # label, fault_detail, load_nm, 4 vibration, 5 tdms, 5 validity
        ("operacao_normal", "normal", 0, 100, -100, 50, -50, 1000, 1000, 200, 200, 200, True, True, True, True, True),
        ("operacao_normal", "normal", 0, 110, -90, 60, -40, 1100, 1000, 210, 190, 210, True, True, True, True, True),
        ("desgaste_rolamento", "BPFO_03", 0, 300, -300, 150, -150, 1000, 1000, 500, 0, 0, True, True, True, False, False),
        ("desgaste_rolamento", "BPFO_03", 2, 320, -310, 160, -140, 1010, 990, 510, 0, 0, True, True, True, False, False),
    ]
    fixture_df = pd.DataFrame(rows, columns=columns)
    con = duckdb.connect()
    con.register("fixture_df", fixture_df)
    con.execute(f"create table {TABLE_NAME} as select * from fixture_df")
    return con


def test_label_distribution_counts_rows_per_class(synthetic_con):
    df = eda.label_distribution(synthetic_con)
    counts = dict(zip(df["label"], df["rows"]))
    assert counts["operacao_normal"] == 2
    assert counts["desgaste_rolamento"] == 2


def test_validity_by_fault_flags_missing_channels(synthetic_con):
    df = eda.validity_by_fault(synthetic_con)
    bpfo_row = df[df["fault_detail"] == "BPFO_03"].iloc[0]
    assert bpfo_row["corrente_fase_v_valida"] == 0.0
    assert bpfo_row["corrente_fase_w_valida"] == 0.0
    assert bpfo_row["corrente_fase_u_valida"] == 1.0

    normal_row = df[df["fault_detail"] == "normal"].iloc[0]
    assert normal_row["corrente_fase_v_valida"] == 1.0


def test_channel_stats_by_label_excludes_invalid_rows_from_mean(synthetic_con):
    # scale = 32768 makes physical_expr's (raw/32768)*scale an identity, so
    # the fixture's raw values can be read directly as the expected physical
    # values below without extra arithmetic.
    scale_factors = {column: 32768.0 for column in MEASUREMENT_COLUMNS}
    df = eda.channel_stats_by_label(synthetic_con, scale_factors)
    bpfo_row = df[df["label"] == "desgaste_rolamento"].iloc[0]
    # corrente_fase_v is always 0/invalid for BPFO in the fixture; the mean
    # of an all-filtered-out aggregate is NULL, not 0.0 (would corrupt the
    # stat if it were silently treated as a real zero reading).
    assert pd.isna(bpfo_row["corrente_fase_v_mean"])


def test_fit_zscore_params_normalizes_to_zero_mean(synthetic_con):
    # scale = 32768 makes physical_expr's (raw/32768)*scale an identity, so
    # the fixture's raw values can be read directly as the expected physical
    # values below without extra arithmetic.
    scale_factors = {column: 32768.0 for column in MEASUREMENT_COLUMNS}
    params = fit_zscore_params(synthetic_con, scale_factors)
    normal_signal = params["aceleracao_x_mancal_a"]
    assert normal_signal.mean == pytest.approx((100 + 110 + 300 + 320) / 4)
    normalized = normal_signal.apply([normal_signal.mean])
    assert normalized[0] == pytest.approx(0.0)
