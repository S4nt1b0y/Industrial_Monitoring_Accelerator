"""Q8.8 fixed-point helpers shared by all matrix_inv reference algorithms.

I/O words are 16-bit signed Q8.8; MAC/division
intermediates are kept as 32-bit Q16.16 (the natural width of a 16x16
signed product) and only rescaled/saturated back to Q8.8 when written to
an I/O register.
"""

import numpy as np

FRAC_BITS = 8
WORD_BITS = 16
ACC_FRAC_BITS = 2 * FRAC_BITS
ACC_BITS = 2 * WORD_BITS

Q88_MIN = -(1 << (WORD_BITS - 1))
Q88_MAX = (1 << (WORD_BITS - 1)) - 1
ACC_MIN = -(1 << (ACC_BITS - 1))
ACC_MAX = (1 << (ACC_BITS - 1)) - 1

# Wider container for cofactors.py's internal cofactor/subdeterminant
# signals: same FRAC_BITS (8) as Q8.8, just more integer headroom, so a
# well-conditioned matrix's minors don't saturate at every multiply the
# way plain Q8.8 does at N=4.
WIDE_BITS = 32
WIDE_MIN = -(1 << (WIDE_BITS - 1))
WIDE_MAX = (1 << (WIDE_BITS - 1)) - 1

# Singularity threshold: 4 LSB of Q8.8.
EPSILON_Q88 = 4

def _bit_range(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

def quantize(x, saturate=True):
    """Float array/scalar -> Q8.8 raw int16 (as int64 container)."""
    raw = np.round(np.asarray(x, dtype=np.float64) * (1 << FRAC_BITS)).astype(np.int64)
    if saturate:
        raw = np.clip(raw, Q88_MIN, Q88_MAX)
    return raw

def dequantize(raw):
    """Q8.8 raw int -> float."""
    return np.asarray(raw, dtype=np.float64) / (1 << FRAC_BITS)

def widen_mul(a_raw, b_raw):
    """Q8.8 x Q8.8 -> Q16.16 accumulator (int64 container, 32-bit hardware width)."""
    return np.asarray(a_raw, dtype=np.int64) * np.asarray(b_raw, dtype=np.int64)

def rescale_acc_to_q88(acc_raw, saturate=True):
    """Q16.16 accumulator -> Q8.8 raw, round-to-nearest then saturate."""
    acc = np.asarray(acc_raw, dtype=np.int64)
    # Round-to-nearest before the arithmetic right shift (floor division
    # would bias negative results toward -inf).
    rounded = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
    if saturate:
        rounded = np.clip(rounded, Q88_MIN, Q88_MAX)
    return rounded

def is_pivot_singular(pivot_raw, epsilon=EPSILON_Q88):
    return bool(abs(int(pivot_raw)) < epsilon)

def q88_mul(a_raw, b_raw, bits=WORD_BITS):
    """Scalar Qa.b x Qa.b -> same format, rounded and saturated (single MAC term).

    `bits` selects the container width of both operands and the result
    (16 for plain Q8.8, WIDE_BITS for cofactors.py's internal signals);
    FRAC_BITS (8) is always the same. Bit-exact match for
    common/q88_ops.vh:q88_mul (04.RTL/matrix_inv).
    """
    lo, hi = _bit_range(bits)
    acc = int(a_raw) * int(b_raw)
    rounded = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
    return int(np.clip(rounded, lo, hi))

def q88_add_sat(a_raw, b_raw, bits=WORD_BITS):
    lo, hi = _bit_range(bits)
    return int(np.clip(int(a_raw) + int(b_raw), lo, hi))

def q88_sub_sat(a_raw, b_raw, bits=WORD_BITS):
    lo, hi = _bit_range(bits)
    return int(np.clip(int(a_raw) - int(b_raw), lo, hi))

def q88_neg_sat(a_raw, bits=WORD_BITS):
    lo, hi = _bit_range(bits)
    return int(np.clip(-int(a_raw), lo, hi))

def q88_divide(a_raw, b_raw):
    """Q8.8 / Q8.8 -> Q8.8, floor(|a| * 2**FRAC_BITS / |b|) with sign, saturated.

    Bit-exact match for the restoring shift-subtract divider
    (common/divider_q88.v): unsigned magnitudes only, truncating (no
    rounding) integer division.
    """
    a_raw = int(a_raw)
    b_raw = int(b_raw)
    sign = -1 if (a_raw < 0) != (b_raw < 0) else 1
    quotient = (abs(a_raw) << FRAC_BITS) // abs(b_raw)
    quotient *= sign
    return int(np.clip(quotient, Q88_MIN, Q88_MAX))
