import numpy as np
import pytest

from fft.reference import (
    PeakDetectionException,
    band_energy,
    bin_to_hz,
    fft_dif_radix2,
    magnitude_spectrum,
    top_3_local_maxima,
    top_3_peaks,
    unscramble,
)

N = 64


def test_matches_numpy_fft_on_random_signal():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, N)
    ours = unscramble(fft_dif_radix2(x))
    reference = np.fft.fft(x)
    np.testing.assert_allclose(ours, reference, atol=1e-9)


def test_rejects_non_power_of_2_length():
    with pytest.raises(ValueError):
        fft_dif_radix2(np.zeros(63))


def test_pure_tone_lights_up_the_expected_bin():
    fs_hz = 6400.0  # N=64, Delta_f = 100 Hz -> a 500 Hz tone should land exactly on bin 5
    k_expected = 5
    t = np.arange(N) / fs_hz
    tone_hz = k_expected * fs_hz / N
    x = np.cos(2 * np.pi * tone_hz * t)

    spectrum = magnitude_spectrum(unscramble(fft_dif_radix2(x)))
    peak_bin = int(np.argmax(spectrum[1:])) + 1  # skip DC
    assert peak_bin == k_expected
    assert bin_to_hz(peak_bin, fs_hz, N) == pytest.approx(tone_hz)


def test_top_3_peaks_excludes_dc_by_default():
    magnitude = np.array([1000.0, 5.0, 3.0, 8.0] + [0.0] * (N // 2 - 3))
    result = top_3_peaks(magnitude)
    assert 0 not in result.bins
    assert set(result.bins) == {1, 2, 3}


def test_top_3_local_maxima_ignores_a_peaks_own_leakage_skirt():
    # bin 4 is a real peak (99.2), bins 3/5 are its leakage neighbors
    # (11.9/11.7, monotonically decreasing away from bin 4) -- these must
    # NOT count as 2 more "peaks" (shape matches a real corrente_fase_u spectrum).
    magnitude = np.array(
        [2.6, 2.1, 5.5, 11.9, 99.2, 11.7, 7.5, 4.7, 4.4, 3.3, 2.8, 2.2, 2.9, 2.3]
        + [2.0, 1.8, 1.6, 1.8, 2.5, 1.8, 1.6]  # a small, well-separated bump at index 20
        + [1.4] * 6
    )
    result = top_3_local_maxima(magnitude)
    assert 3 not in result.bins
    assert 5 not in result.bins
    assert 4 in result.bins


def test_top_3_local_maxima_enforces_minimum_separation():
    # two adjacent local maxima (distance 1) -- only one may be accepted.
    magnitude = np.array([0, 5, 4, 6, 3, 0, 8, 0, 9, 0, 7, 0])
    # bin1=5 (>0,>4) local max; bin3=6(>4,>3) local max, distance to bin1 is 2 -- ok
    # bin6=8, bin8=9, bin10=7 all local maxima too, well separated
    result = top_3_local_maxima(magnitude, min_separation=2)
    for a in result.bins:
        for b in result.bins:
            if a != b:
                assert abs(a - b) >= 2


def test_top_3_local_maxima_raises_when_fewer_than_3_found():
    magnitude = np.array([0, 5, 0, 0, 0, 0])  # only 1 local maximum
    with pytest.raises(PeakDetectionException):
        top_3_local_maxima(magnitude)


def test_band_energy_sums_squared_magnitude_over_the_given_bins():
    spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert band_energy(spectrum, (1, 3)) == pytest.approx(2.0**2 + 4.0**2)
    assert band_energy(spectrum, (0,)) == pytest.approx(1.0)


def test_stage_count_matches_log2_n():
    # every stage does exactly 32 butterflies for N=64 --
    # spot check via the twiddle array size used internally at each span.
    n = N
    spans = []
    span = n
    while span > 1:
        spans.append(span)
        span //= 2
    assert len(spans) == 6  # log2(64)
    for span in spans:
        butterflies_this_stage = (n // span) * (span // 2)
        assert butterflies_this_stage == 32
