#!/usr/bin/env python3
"""Fixed-point Q1.15 reference for the 64-point DIF FFT RTL."""

import math
from pathlib import Path

N = 64
M = 6
W = 16
FRACW = 15
NUM_WINDOWS = 10
INPUT_FILE = Path("input.txt")
OUT_FILE = Path("python_output.txt")


def to_signed(value, bits):
    mask = (1 << bits) - 1
    value &= mask
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def q15_saturate(value):
    return max(-32768, min(32767, int(value)))


def q15_twiddle(k):
    scale = 1 << FRACW
    re = q15_saturate(round(math.cos(-2.0 * math.pi * k / N) * scale))
    im = q15_saturate(round(math.sin(-2.0 * math.pi * k / N) * scale))
    return re, im

def read_input_file(path=INPUT_FILE):
    if not path.exists():
        raise ValueError(
            f"{path}: not found. Run this script from 05.Sim/fft or add input.txt there."
        )

    windows = [[None for _ in range(N)] for _ in range(NUM_WINDOWS)]
    for line_no, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue

        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_no}: expected 4 fields")

        window, sample, re, im = map(int, fields)
        if not (0 <= window < NUM_WINDOWS):
            raise ValueError(f"{path}:{line_no}: window out of range")
        if not (0 <= sample < N):
            raise ValueError(f"{path}:{line_no}: sample out of range")
        if windows[window][sample] is not None:
            raise ValueError(f"{path}:{line_no}: duplicate sample")

        windows[window][sample] = (to_signed(re, W), to_signed(im, W))

    for window in range(NUM_WINDOWS):
        for sample in range(N):
            if windows[window][sample] is None:
                raise ValueError(f"{path}: missing window {window} sample {sample}")

    return windows


def brev(value, bits=M):
    out = 0
    for _ in range(bits):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def rtl_shift_slice(value, source_bits, high, low):
    """Replicate Verilog signed slice assignment, e.g. sum[16:1]."""
    unsigned = value & ((1 << source_bits) - 1)
    sliced = (unsigned >> low) & ((1 << (high - low + 1)) - 1)
    return to_signed(sliced, high - low + 1)


def butterfly(a_re, a_im, b_re, b_im, w_re, w_im):
    sum_re = to_signed(a_re + b_re, W + 1)
    sum_im = to_signed(a_im + b_im, W + 1)
    diff_re = to_signed(a_re - b_re, W + 1)
    diff_im = to_signed(a_im - b_im, W + 1)

    x0_re = rtl_shift_slice(sum_re, W + 1, W, 1)
    x0_im = rtl_shift_slice(sum_im, W + 1, W, 1)

    pr = to_signed(diff_re * w_re - diff_im * w_im, 2 * W + 2)
    pi = to_signed(diff_re * w_im + diff_im * w_re, 2 * W + 2)
    x1_re = rtl_shift_slice(pr, 2 * W + 2, W + FRACW, FRACW + 1)
    x1_im = rtl_shift_slice(pi, 2 * W + 2, W + FRACW, FRACW + 1)

    return x0_re, x0_im, x1_re, x1_im


def fft_dif_q15(in_re, in_im):
    src_re = [to_signed(x, W) for x in in_re]
    src_im = [to_signed(x, W) for x in in_im]

    for stage in range(M):
        dst_re = [0] * N
        dst_im = [0] * N
        span = 1 << ((M - 1) - stage)

        for j in range(N // 2):
            pos = j & (span - 1)
            group = j >> ((M - 1) - stage)
            base = group << ((M - 1) - stage + 1)
            a_idx = base + pos
            b_idx = a_idx + span
            k = pos << stage
            w_re, w_im = q15_twiddle(k)

            x0_re, x0_im, x1_re, x1_im = butterfly(
                src_re[a_idx], src_im[a_idx],
                src_re[b_idx], src_im[b_idx],
                w_re, w_im,
            )
            dst_re[a_idx] = x0_re
            dst_im[a_idx] = x0_im
            dst_re[b_idx] = x1_re
            dst_im[b_idx] = x1_im

        src_re = dst_re
        src_im = dst_im

    out_re = [0] * N
    out_im = [0] * N
    for i in range(N):
        out_re[i] = src_re[brev(i)]
        out_im[i] = src_im[brev(i)]

    return out_re, out_im


def main():
    windows = read_input_file()

    with OUT_FILE.open("w", encoding="ascii") as file:
        for window, samples in enumerate(windows):
            in_re = [re for re, _ in samples]
            in_im = [im for _, im in samples]
            out_re, out_im = fft_dif_q15(in_re, in_im)

            for idx, (re, im) in enumerate(zip(out_re, out_im)):
                file.write(f"{window} {idx} {re} {im}\n")

    print(f"Python fixed-point FFT output written to {OUT_FILE}")


if __name__ == "__main__":
    main()
