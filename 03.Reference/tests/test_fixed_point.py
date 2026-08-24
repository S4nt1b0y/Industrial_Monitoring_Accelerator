import numpy as np

from matrix_inv.fixed_point import (
    EPSILON_Q88,
    Q88_MAX,
    Q88_MIN,
    dequantize,
    is_pivot_singular,
    quantize,
    rescale_acc_to_q88,
    widen_mul,
)

HALF_LSB = 1.0 / 256 / 2


def test_quantize_dequantize_round_trip():
    x = np.array([0.0, 1.0, -1.0, 3.5, -3.5, 127.99, -128.0])
    raw = quantize(x)
    back = dequantize(raw)
    assert np.all(np.abs(back - x) <= HALF_LSB + 1e-9)


def test_quantize_saturates():
    raw = quantize(np.array([1000.0, -1000.0]))
    assert raw[0] == Q88_MAX
    assert raw[1] == Q88_MIN


def test_widen_mul_rescale_round_trip():
    a = quantize(np.array([2.0]))
    b = quantize(np.array([3.0]))
    acc = widen_mul(a, b)
    result = rescale_acc_to_q88(acc)
    assert dequantize(result)[0] == 6.0


def test_widen_mul_rescale_rounding():
    # 0.5 * 0.5 = 0.25 -> exact in Q8.8 (raw 64), no rounding needed.
    a = quantize(np.array([0.5]))
    b = quantize(np.array([0.5]))
    acc = widen_mul(a, b)
    result = rescale_acc_to_q88(acc)
    assert dequantize(result)[0] == 0.25


def test_is_pivot_singular_boundary():
    assert is_pivot_singular(0)
    assert is_pivot_singular(EPSILON_Q88 - 1)
    assert not is_pivot_singular(EPSILON_Q88)
    assert is_pivot_singular(-(EPSILON_Q88 - 1))
    assert not is_pivot_singular(-EPSILON_Q88)
