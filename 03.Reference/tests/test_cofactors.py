import numpy as np
import pytest

from matrix_inv.algorithms.cofactors import det_fixed, det_float, solve_fixed, solve_float
from matrix_inv.algorithms.gauss_jordan import solve_fixed as gj_solve_fixed
from matrix_inv.fixed_point import dequantize, quantize

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
def test_det_fixed_matches_float_oracle(n):
    A, _ = WELL_CONDITIONED[n]
    a_raw = quantize(A)
    det_raw = det_fixed(a_raw, n)
    assert abs(det_raw / 256.0 - det_float(A)) < _tolerance(n) * n


@pytest.mark.parametrize("n", [2, 3, 4])
def test_agrees_with_gauss_jordan(n):
    # Different computational path (no pivoting, all multiplication), same
    # exact-arithmetic answer -- fixed-point rounding differs, but both
    # should land close to each other (and to the float oracle) for a
    # well-conditioned matrix.
    A, b = WELL_CONDITIONED[n]
    a_raw, b_raw = quantize(A), quantize(b)

    x_cof, ainv_cof, singular_cof = solve_fixed(a_raw, b_raw, n)
    x_gj, ainv_gj, singular_gj = gj_solve_fixed(a_raw, b_raw, n)
    assert not singular_cof and not singular_gj

    tol = _tolerance(n) * 2  # two independent rounding paths
    assert np.max(np.abs(dequantize(x_cof) - dequantize(x_gj))) < tol
    assert np.max(np.abs(dequantize(ainv_cof) - dequantize(ainv_gj))) < tol


@pytest.mark.parametrize("n", [2, 3, 4])
def test_exactly_singular_matrix_flags_singular(n):
    A = np.zeros((n, n))
    A[0, :] = 1.0
    if n > 1:
        A[1, :] = 2.0
    b = np.ones(n)

    _, _, singular = solve_fixed(quantize(A), quantize(b), n)
    assert singular


def test_2x2_matches_textbook_formula():
    # A^-1 = 1/det * [[d,-b],[-c,a]]
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_raw, ainv_raw, singular = solve_fixed(quantize(A), quantize(b), 2)
    assert not singular
    ainv = dequantize(ainv_raw)
    assert abs(ainv[0, 0] - 0.3) < 1 / 256
    assert abs(ainv[0, 1] - (-0.1)) < 1 / 256
    assert abs(ainv[1, 0] - (-0.2)) < 1 / 256
    assert abs(ainv[1, 1] - 0.4) < 1 / 256
