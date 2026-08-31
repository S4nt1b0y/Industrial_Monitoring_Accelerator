"""Gauss-Jordan elimination with partial pivoting, augmented [A | I | b].

Bit-true fixed-point model mirrors the planned RTL datapath 1:1 so the
generated .hex vectors are valid golden references for the testbench:
  - normalize step: whole pivot row divided by the pivot, one Q8.8/Q8.8
    division per column, via fixed_point.q88_divide (matches the
    shift-subtract restoring divider).
  - eliminate step: for every other row, row_i -= factor * row_pivot
    across the whole augmented row, using the shared widen_mul/
    rescale_acc_to_q88 MAC path.
  - Both are 1:1 with 04.RTL/matrix_inv/matrix_inv_gj.v.
"""

import numpy as np

from matrix_inv.fixed_point import (
    Q88_MAX,
    Q88_MIN,
    FRAC_BITS,
    dequantize,
    is_pivot_singular,
    q88_divide,
    quantize,
    rescale_acc_to_q88,
    widen_mul,
)

ONE_Q88 = 1 << FRAC_BITS


def solve_float(A, b):
    """Floating-point oracle (numpy), for cross-checking the fixed-point path."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[0]
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None, None, True
    a_inv = np.linalg.inv(A)
    x = a_inv @ b
    return x, a_inv, False


def solve_fixed(a_raw, b_raw, n):
    """Fixed-point Q8.8 Gauss-Jordan, bit-true to the RTL datapath.

    a_raw: (n, n) int array, Q8.8 raw values.
    b_raw: (n,) int array, Q8.8 raw values.
    Returns (x_raw, ainv_raw, singular).
    """
    m = 2 * n + 1
    aug = np.zeros((n, m), dtype=np.int64)
    aug[:, 0:n] = np.asarray(a_raw, dtype=np.int64)
    for i in range(n):
        aug[i, n + i] = ONE_Q88
    aug[:, 2 * n] = np.asarray(b_raw, dtype=np.int64)

    for k in range(n):
        p = k + int(np.argmax(np.abs(aug[k:, k])))
        pivot = int(aug[p, k])
        if is_pivot_singular(pivot):
            return None, None, True
        if p != k:
            aug[[k, p], :] = aug[[p, k], :]
            pivot = int(aug[k, k])

        for col in range(m):
            aug[k, col] = q88_divide(int(aug[k, col]), pivot)

        for i in range(n):
            if i == k:
                continue
            factor = int(aug[i, k])
            if factor == 0:
                continue
            products = widen_mul(np.full(m, factor, dtype=np.int64), aug[k, :])
            delta = rescale_acc_to_q88(products)
            aug[i, :] = np.clip(aug[i, :] - delta, Q88_MIN, Q88_MAX)

    ainv_raw = aug[:, n:2 * n]
    x_raw = aug[:, 2 * n]
    return x_raw, ainv_raw, False
