"""Radix-2 decimation-in-frequency (DIF) FFT, iterative and organized by
stage so its operation/cycle count can be measured directly instead of
reasoned about from a recursive form: 6 stages (log2(64)), 32
butterflies per stage, output in bit-reversed order.

Floating-point only -- this is the "ideal"/oracle reference; fixed-point
quantization is a hardware-side concern, not modeled here.
"""

from dataclasses import dataclass

import numpy as np


def fft_dif_radix2(x):
    """Returns the DFT in bit-reversed order (same convention as the RTL)."""
    x = np.array(x, dtype=complex)
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError(f"length must be a power of 2, got {n}")

    span = n
    while span > 1:
        half = span // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / span)
        for group_start in range(0, n, span):
            # top/bottom are views into x -- copy before the first
            # assignment overwrites data the second one still needs.
            top = x[group_start : group_start + half].copy()
            bottom = x[group_start + half : group_start + span].copy()
            x[group_start : group_start + half] = top + bottom
            x[group_start + half : group_start + span] = (top - bottom) * twiddle
        span = half
    return x


def bit_reverse_indices(n):
    bits = int(np.log2(n))
    return np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)])


def unscramble(x_bit_reversed):
    """Reorders a bit-reversed-order FFT result into natural bin order.

    X_true[k] = x_bit_reversed[bitreverse(k)] (bit-reversal is self-inverse).
    """
    return x_bit_reversed[bit_reverse_indices(len(x_bit_reversed))]


def magnitude_spectrum(x_natural_order):
    """|X[k]| for the usable bins (0..N/2) of a real-valued input's FFT."""
    n = len(x_natural_order)
    return np.abs(x_natural_order[: n // 2 + 1])


def bin_to_hz(k, fs_hz, n_fft):
    return k * fs_hz / n_fft


def band_energy(spectrum, bins):
    """Sum of squared magnitude over a set of bins (a spectral energy
    feature). Which bins to pass is signal-specific -- see
    dataset/bearing_bands.py for this project's bearing-fault bands."""
    return float(np.sum(np.asarray(spectrum)[list(bins)] ** 2))


@dataclass(frozen=True)
class PeakDetection:
    bins: tuple
    magnitudes: tuple


def top_3_peaks(magnitude, exclude_dc=True):
    """3 largest-magnitude bins.

    Not suitable for signals with one dominant peak and no real harmonic
    structure: a strong peak's own spectral-leakage skirt (its immediate
    neighbor bins) gets counted as 2 extra "peaks", and adjacent bin
    indices are always coprime -- a GCD-based estimator downstream would
    then always see 1. Use top_3_local_maxima below for that case.
    """
    usable = magnitude.copy()
    if exclude_dc:
        usable[0] = -np.inf
    top_indices = tuple(sorted(int(k) for k in np.argsort(usable)[-3:]))
    top_mags = tuple(float(magnitude[k]) for k in top_indices)
    return PeakDetection(bins=top_indices, magnitudes=top_mags)


class PeakDetectionException(Exception):
    """Fewer than 3 qualifying local maxima found -- drop the window."""


def top_3_local_maxima(magnitude, min_separation=2, exclude_dc=True):
    """3 distinct local maxima (magnitude[k] > both neighbors), greedily
    accepted by descending magnitude with a minimum bin separation.

    `min_separation=2` is the smallest value that excludes a peak's own
    immediate leakage neighbors (distance 1) while still allowing
    closely-spaced but genuinely distinct harmonics.
    """
    n = len(magnitude)
    start = 1 if exclude_dc else 0
    candidates = [
        k
        for k in range(start, n - 1)
        if magnitude[k] > magnitude[k - 1] and magnitude[k] > magnitude[k + 1]
    ]
    candidates.sort(key=lambda k: -magnitude[k])

    accepted = []
    for k in candidates:
        if all(abs(k - a) >= min_separation for a in accepted):
            accepted.append(k)
        if len(accepted) == 3:
            break

    if len(accepted) < 3:
        raise PeakDetectionException(
            f"only {len(accepted)} qualifying local maxima found, need 3"
        )

    accepted.sort()
    return PeakDetection(
        bins=tuple(accepted), magnitudes=tuple(float(magnitude[k]) for k in accepted)
    )
