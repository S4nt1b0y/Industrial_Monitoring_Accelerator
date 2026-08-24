import numpy as np
import pytest

from matrix_inv.algorithms.gauss_jordan import solve_fixed, solve_float
from matrix_inv.fixed_point import dequantize, q88_divide, quantize

WELL_CONDITIONED = {
    2: (np.array([[4.0, 1.0], [2.0, 3.0]]), np.array([1.0, 2.0])),
    3: (
        np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]),
        np.array([1.0, 2.0, 3.0]),
    ),
    4: (
        np.array(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        ),
        np.array([1.0, 2.0, 3.0, 4.0]),
    ),
}


def _tolerance(n):
    # A couple of LSB per elimination step; error accumulates with n.
    return 4 * n / 256.0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_well_conditioned_matches_float_oracle(n):
    A, b = WELL_CONDITIONED[n]
    x_ref, ainv_ref, singular_ref = solve_float(A, b)
    assert not singular_ref

    x_raw, ainv_raw, singular = solve_fixed(quantize(A), quantize(b), n)
    assert not singular

    tol = _tolerance(n)
    assert np.max(np.abs(dequantize(x_raw) - x_ref)) < tol
    assert np.max(np.abs(dequantize(ainv_raw) - ainv_ref)) < tol


@pytest.mark.parametrize("n", [2, 3, 4])
def test_exactly_singular_matrix_flags_singular(n):
    A = np.zeros((n, n))
    A[0, :] = 1.0
    if n > 1:
        A[1, :] = 2.0  # duplicate of row 0 (scaled) -> singular
    b = np.ones(n)

    _, _, singular = solve_fixed(quantize(A), quantize(b), n)
    assert singular


def test_near_singular_matrix_flags_singular_after_quantization():
    # det = 1*4.001 - 2*2 = 0.001, below Q8.8 resolution (1/256 ~ 0.0039):
    # quantization collapses the rows to (near-)proportional, tripping the
    # pivot epsilon even though the float matrix is technically invertible.
    A = np.array([[1.0, 2.0], [2.0, 4.001]])
    b = np.array([1.0, 2.0])

    _, _, singular = solve_fixed(quantize(A), quantize(b), 2)
    assert singular


def test_q88_divide_self_division_is_exact_one():
    for raw in (256, -256, 512, 4, -4, 32767):
        assert q88_divide(raw, raw) == 256


def test_q88_divide_saturates():
    from matrix_inv.fixed_point import Q88_MAX

    assert q88_divide(32767, 4) == Q88_MAX
