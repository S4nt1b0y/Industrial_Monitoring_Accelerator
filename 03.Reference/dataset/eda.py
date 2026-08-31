"""Descriptive statistics over the consolidated motor-measurements dataset."""

from .loader import MEASUREMENT_COLUMNS, TABLE_NAME, VALIDITY_COLUMNS, physical_expr


def class_distribution(con, table=TABLE_NAME):
    """Row count per (label, fault_detail, load_nm) — the finest-grained grouping."""
    return con.execute(f"""
        select label, fault_detail, load_nm, count(*) as rows
        from {table}
        group by label, fault_detail, load_nm
        order by label, fault_detail, load_nm
    """).df()


def label_distribution(con, table=TABLE_NAME):
    """Row count per label (the 4 target classes)."""
    return con.execute(f"""
        select label, count(*) as rows, count(distinct fault_detail || '|' || load_nm) as n_sources
        from {table}
        group by label
        order by rows desc
    """).df()


def validity_by_fault(con, table=TABLE_NAME):
    """Fraction of rows where each TDMS-derived channel is real (not a missing sensor).

    A column consistently at 0.0 for a given fault_detail means that
    channel was never recorded for that source, not that the physical
    quantity was zero.
    """
    validity_means = ", ".join(f"avg({column}::DOUBLE) as {column}" for column in VALIDITY_COLUMNS)
    return con.execute(f"""
        select label, fault_detail, {validity_means}
        from {table}
        group by label, fault_detail
        order by label, fault_detail
    """).df()


def channel_stats_by_label(con, scale_factors, table=TABLE_NAME):
    """Physical-unit mean/std/RMS per channel per label.

    Uses only rows where the channel's own validity flag is true (where a
    flag exists) so a channel a fault class never recorded doesn't drag its
    stats toward zero.
    """
    selects = []
    for column in MEASUREMENT_COLUMNS:
        expr = physical_expr(column, scale_factors[column])
        valid_column = f"{column}_valida"
        has_validity = valid_column in _validity_column_names()
        filt = f" filter (where {valid_column})" if has_validity else ""
        selects.append(f"avg({expr}){filt} as {column}_mean")
        selects.append(f"stddev_samp({expr}){filt} as {column}_std")
        selects.append(f"sqrt(avg(({expr})*({expr})){filt}) as {column}_rms")
    select_clause = ",\n        ".join(selects)
    return con.execute(f"""
        select label,
        {select_clause}
        from {table}
        group by label
        order by label
    """).df()


def _validity_column_names():
    return set(VALIDITY_COLUMNS)
