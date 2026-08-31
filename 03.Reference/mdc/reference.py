"""Euclidean GCD-based fundamental-frequency estimator (MDC module).

Bit-true to the planned RTL datapath: "a mod b" is computed by repeated
subtraction (a -= b while a >= b), not Python's %, because the hardware
has a subtractor and comparator, no divider. gcd_by_subtraction's cycle
count is a direct, literal estimate for that datapath (1 cycle per
subtraction, 1 per swap).
"""

from dataclasses import dataclass

from fft.reference import bin_to_hz

DOMINANCE_RATIO_THRESHOLD = 5.0


class MDCException(Exception):
    """Invalid/unreliable input. Hardware would assert an invalid signal
    and drop the window instead of raising -- callers that want that
    behavior should catch this and treat it as "resultado_valido = 0"."""


def gcd_by_subtraction(a, b):
    """Returns (gcd, cycles). Mirrors the RTL: subtractor + comparator only."""
    cycles = 0
    while b != 0:
        while a >= b:
            a -= b
            cycles += 1
        a, b = b, a
        cycles += 1  # swap/register-update cycle
    return a, cycles


@dataclass(frozen=True)
class FundamentalEstimate:
    k0: int
    f0_hz: float
    cycles: int


def estimate_fundamental(a, b, c, fs_hz, n_fft, k0_min=1):
    """a, b, c: the 3 peak bin indices from the peak detector (0 <= index < n_fft)."""
    if a == 0 or b == 0 or c == 0:
        raise MDCException(f"bin index 0 (DC) among peaks: a={a}, b={b}, c={c}")

    m, cycles_1 = gcd_by_subtraction(a, b)
    k0, cycles_2 = gcd_by_subtraction(m, c)
    cycles = cycles_1 + cycles_2

    if k0 < k0_min:
        raise MDCException(f"k0={k0} below configured minimum {k0_min}")

    f0_hz = k0 * fs_hz / n_fft
    return FundamentalEstimate(k0=k0, f0_hz=f0_hz, cycles=cycles)


def estimate_fundamental_from_peaks(peaks, fs_hz, n_fft, k0_min=1):
    """f0 estimate from a fft.reference.PeakDetection, with a dominance
    fallback around the 3-peak GCD.

    GCD-based estimation assumes a fundamental that is weaker than its
    own harmonics. A signal with one dominant peak and no real harmonic
    structure above the noise floor breaks that assumption: forcing 3
    peaks through GCD picks up noise-floor bins at effectively random
    positions alongside the real peak. When the strongest peak clearly
    dominates the second-strongest (ratio >= DOMINANCE_RATIO_THRESHOLD),
    its own bin is used directly; otherwise a real harmonic structure
    may be present, and the module falls through to the GCD estimate.
    """
    sorted_mags = sorted(peaks.magnitudes, reverse=True)
    second_strongest = sorted_mags[1] if len(sorted_mags) > 1 else 0.0
    if second_strongest > 0 and sorted_mags[0] / second_strongest >= DOMINANCE_RATIO_THRESHOLD:
        dominant_bin = peaks.bins[peaks.magnitudes.index(sorted_mags[0])]
        if dominant_bin < k0_min:
            raise MDCException(f"dominant bin {dominant_bin} below configured minimum {k0_min}")
        return FundamentalEstimate(
            k0=dominant_bin, f0_hz=bin_to_hz(dominant_bin, fs_hz, n_fft), cycles=1
        )
    return estimate_fundamental(*peaks.bins, fs_hz, n_fft, k0_min=k0_min)
