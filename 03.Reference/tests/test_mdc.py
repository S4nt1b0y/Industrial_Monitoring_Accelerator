import pytest

from fft.reference import PeakDetection
from mdc.reference import (
    MDCException,
    estimate_fundamental,
    estimate_fundamental_from_peaks,
    gcd_by_subtraction,
)

FS_HZ = 25_600.0
N_FFT = 64


def test_enunciado_worked_example():
    # Enunciado Secao 3.1: peaks at bins 12, 18, 30 -> k0 = MDC(12,18,30) = 6.
    result = estimate_fundamental(12, 18, 30, FS_HZ, N_FFT)
    assert result.k0 == 6
    assert result.f0_hz == pytest.approx(6 * FS_HZ / N_FFT)


@pytest.mark.parametrize("a,b,c", [(0, 18, 30), (12, 0, 30), (12, 18, 0)])
def test_zero_bin_is_rejected(a, b, c):
    with pytest.raises(MDCException):
        estimate_fundamental(a, b, c, FS_HZ, N_FFT)


def test_k0_below_minimum_is_rejected():
    # gcd(7, 11, 13) = 1: coprime bins, k0=1 falls below a k0_min=2 policy.
    with pytest.raises(MDCException):
        estimate_fundamental(7, 11, 13, FS_HZ, N_FFT, k0_min=2)


def test_k0_at_minimum_is_accepted():
    result = estimate_fundamental(7, 11, 13, FS_HZ, N_FFT, k0_min=1)
    assert result.k0 == 1


def test_gcd_by_subtraction_matches_python_gcd():
    import math

    for a, b in [(12, 18), (7, 11), (63, 1), (48, 18), (1, 1)]:
        got, _ = gcd_by_subtraction(a, b)
        assert got == math.gcd(a, b)


def test_worst_case_cycle_count_matches_the_theoretical_estimate():
    # Theoretical worst case for two gcd calls with 6-bit inputs is ~126
    # cycles (b=1 forces ~63 subtractions per call). Confirm the model
    # actually hits that.
    _, cycles_1 = gcd_by_subtraction(63, 1)
    _, cycles_2 = gcd_by_subtraction(63, 1)
    assert cycles_1 + cycles_2 <= 130
    assert cycles_1 + cycles_2 >= 120


def test_typical_case_is_much_cheaper_than_worst_case():
    result = estimate_fundamental(12, 18, 30, FS_HZ, N_FFT)
    assert result.cycles < 20


def test_dominance_fallback_trusts_a_single_clear_peak_directly():
    # bin 4 dominates 12 and 20 by >5x -- reads off bin 4 directly, does
    # NOT take gcd(4, 12, 20) (which would coincidentally also be 4 here,
    # but the point is the fallback bypasses GCD entirely in this regime).
    peaks = PeakDetection(bins=(4, 12, 20), magnitudes=(99.2, 2.9, 1.7))
    result = estimate_fundamental_from_peaks(peaks, FS_HZ, N_FFT)
    assert result.k0 == 4
    assert result.f0_hz == pytest.approx(4 * FS_HZ / N_FFT)


def test_dominance_fallback_falls_through_to_gcd_when_peaks_are_comparable():
    # no single peak dominates -- genuine multi-harmonic case, use MDC/GCD
    # as the enunciado specifies. gcd(12, 18, 30) = 6 (the worked example).
    peaks = PeakDetection(bins=(12, 18, 30), magnitudes=(100.0, 95.0, 90.0))
    result = estimate_fundamental_from_peaks(peaks, FS_HZ, N_FFT)
    assert result.k0 == 6
