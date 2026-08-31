"""DuckDB access to the consolidated Parquet dataset.

Every function elsewhere in this package takes an open connection plus a
table name (defaulting to TABLE_NAME) instead of hardcoding the real
Parquet path, so tests can register a small synthetic table under the same
name and reuse the exact same SQL.
"""

import csv

import duckdb

from .paths import DATASET_PARQUET, SCALE_REPORT_CSV

TABLE_NAME = "motor_measurements"

VIBRATION_COLUMNS = [
    "aceleracao_x_mancal_a",
    "aceleracao_y_mancal_a",
    "aceleracao_x_mancal_b",
    "aceleracao_y_mancal_b",
]
TDMS_COLUMNS = [
    "temperatura_mancal_a",
    "temperatura_mancal_b",
    "corrente_fase_u",
    "corrente_fase_v",
    "corrente_fase_w",
]
MEASUREMENT_COLUMNS = VIBRATION_COLUMNS + TDMS_COLUMNS
VALIDITY_COLUMNS = [f"{column}_valida" for column in TDMS_COLUMNS]

Q15_SCALE = 32768.0


def open_connection(parquet_path=DATASET_PARQUET, table_name=TABLE_NAME):
    """Open an in-memory DuckDB connection with a view over the Parquet file.

    DuckDB's CREATE VIEW can't take a prepared-statement parameter for the
    path, so the (trusted, project-internal) path is escaped and inlined.
    """
    con = duckdb.connect()
    escaped_path = str(parquet_path).replace("'", "''")
    con.execute(f"create view {table_name} as select * from read_parquet('{escaped_path}')")
    return con


def load_scale_factors(path=SCALE_REPORT_CSV):
    """column -> normalization_factor (max |physical value| seen during extraction).

    Physical value = (q15_raw / 32768) * normalization_factor; see
    build_dataset.py's quantize_q15 for the forward direction this inverts.
    """
    factors = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            factors[row["column"]] = float(row["normalization_factor"])
    return factors


def physical_expr(column, scale):
    """SQL expression that dequantizes a Q1.15 column back to physical units."""
    return f"(CAST({column} AS DOUBLE) / {Q15_SCALE}) * {scale}"
