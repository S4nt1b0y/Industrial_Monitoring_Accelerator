"""Dispatch to the fixed-point solver for a given algorithm."""

from matrix_inv.algorithms import cofactors, gauss_jordan

SOLVERS = {
    "gj": gauss_jordan.solve_fixed,
    "cof": cofactors.solve_fixed,
}

def solve(algo, a_raw, b_raw, n):
    try:
        solver = SOLVERS[algo]
    except KeyError:
        raise ValueError(f"unknown algorithm '{algo}', available: {sorted(SOLVERS)}") from None
    return solver(a_raw, b_raw, n)
