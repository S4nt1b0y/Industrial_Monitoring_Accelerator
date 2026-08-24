"""Cofactor/adjugate method: A^-1 = adj(A) / det(A), adj(A) = cofactor(A)^T.

Bit-true fixed-point model mirrors the planned RTL 1:1
(04.RTL/matrix_inv/common/det_q88.v, matrix_inv_cof.v):
  - det(K) for K==1 is the element itself; for K>=2 it is a row-0 cofactor
    expansion, recursing on (K-1)x(K-1) minors. Every multiply and running
    sum is rounded/saturated individually (fixed_point.q88_mul /
    q88_add_sat) -- no wide accumulator is carried between recursion
    levels, unlike Gauss-Jordan's row MAC.
  - Internal subdeterminant/cofactor signals use WIDE_BITS (32) instead of
    plain Q8.8 (16): even a modest, well-conditioned 4x4 matrix routinely
    produces row-0-expansion terms that overflow Q8.8 (e.g. element 4.0 x
    3x3-minor-determinant 41.0 = 164.0, already past Q8.8's ~128 max). 
    Same FRAC_BITS (8) throughout,
    just more integer headroom; only a/b/x/A_inv stay 16-bit Q8.8.
  - The N^2 cofactors needed for the full adjugate are each an independent
    (N-1)x(N-1) determinant; det(A) itself is read off the row-0
    cofactors, not recomputed separately.
  - The final adj/det division is the same scalar fixed_point.q88_divide
    as Gauss-Jordan's normalize step, just with wide (32-bit) operands --
    q88_divide doesn't hardcode an input width, only the Q8.8 output
    range, so it's unchanged.
"""

import numpy as np

from matrix_inv.fixed_point import (
    WIDE_BITS,
    dequantize,
    is_pivot_singular,
    q88_add_sat,
    q88_divide,
    q88_mul,
    q88_neg_sat,
    quantize,
)

def _minor(a_raw, remove_row, remove_col):
    return np.delete(np.delete(a_raw, remove_row, axis=0), remove_col, axis=1)

def det_fixed(a_raw, k):
    """Determinant of a k x k Q8.8 raw matrix, recursive row-0 expansion.

    Returns a WIDE_BITS-range value for k>=2 (see module docstring); for
    k==1 it is just the original Q8.8 element, trivially within range.
    """
    if k == 1:
        return int(a_raw[0, 0])

    total = None
    for j in range(k):
        minor = _minor(a_raw, 0, j)
        subdet = det_fixed(minor, k - 1)
        signed_subdet = subdet if j % 2 == 0 else q88_neg_sat(subdet, bits=WIDE_BITS)
        term = q88_mul(int(a_raw[0, j]), signed_subdet, bits=WIDE_BITS)
        total = term if total is None else q88_add_sat(total, term, bits=WIDE_BITS)
    return total

def det_float(a):
    return float(np.linalg.det(np.asarray(a, dtype=np.float64)))

def solve_float(A, b):
    """Floating-point oracle (numpy), for cross-checking the fixed-point path."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None, None, True
    a_inv = np.linalg.inv(A)
    x = a_inv @ b
    return x, a_inv, False

def solve_fixed(a_raw, b_raw, n):
    """Fixed-point Q8.8 cofactor/adjugate solve, bit-true to the RTL datapath.

    a_raw: (n, n) int array, Q8.8 raw values.
    b_raw: (n,) int array, Q8.8 raw values.
    Returns (x_raw, ainv_raw, singular).
    """
    a_raw = np.asarray(a_raw, dtype=np.int64)
    b_raw = np.asarray(b_raw, dtype=np.int64)

    cofactor = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            minor = _minor(a_raw, i, j)
            subdet = det_fixed(minor, n - 1)
            cofactor[i, j] = subdet if (i + j) % 2 == 0 else q88_neg_sat(subdet, bits=WIDE_BITS)

    det_raw = None
    for j in range(n):
        term = q88_mul(int(a_raw[0, j]), int(cofactor[0, j]), bits=WIDE_BITS)
        det_raw = term if det_raw is None else q88_add_sat(det_raw, term, bits=WIDE_BITS)

    if is_pivot_singular(det_raw):
        return None, None, True

    ainv_raw = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            ainv_raw[i, j] = q88_divide(int(cofactor[j, i]), det_raw)

    x_raw = np.zeros(n, dtype=np.int64)
    for i in range(n):
        acc = None
        for j in range(n):
            term = q88_mul(int(ainv_raw[i, j]), int(b_raw[j]))
            acc = term if acc is None else q88_add_sat(acc, term)
        x_raw[i] = acc

    return x_raw, ainv_raw, False