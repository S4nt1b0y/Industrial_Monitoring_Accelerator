"""Normalization helpers for downstream feature/training pipelines.

Stats are computed globally (not per label) on purpose: normalizing per
class would leak the target into the features. Compute stats once on the
training split only and reuse the same (mean, std) on val/test — never
refit per split.
"""

from dataclasses import dataclass

import numpy as np

from .loader import MEASUREMENT_COLUMNS, TABLE_NAME, VALIDITY_COLUMNS, physical_expr


@dataclass(frozen=True)
class ZScoreParams:
    mean: float
    std: float

    def apply(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.std == 0.0:
            return values - self.mean
        return (values - self.mean) / self.std

    def invert(self, normalized):
        normalized = np.asarray(normalized, dtype=np.float64)
        return normalized * self.std + self.mean


def fit_zscore_params(con, scale_factors, table=TABLE_NAME, where=None):
    """One ZScoreParams per measurement column, in physical units.

    `where` is an optional raw SQL condition (e.g. a train-split filter on
    fault_detail/load_nm) applied before computing mean/std.
    """
    validity_names = set(VALIDITY_COLUMNS)
    filter_clause = f"where {where}" if where else ""
    params = {}
    for column in MEASUREMENT_COLUMNS:
        expr = physical_expr(column, scale_factors[column])
        valid_column = f"{column}_valida"
        agg_filter = f" filter (where {valid_column})" if valid_column in validity_names else ""
        row = con.execute(f"""
            select avg({expr}){agg_filter} as mean, stddev_samp({expr}){agg_filter} as std
            from {table}
            {filter_clause}
        """).fetchone()
        mean, std = row
        params[column] = ZScoreParams(mean=float(mean or 0.0), std=float(std or 0.0))
    return params
