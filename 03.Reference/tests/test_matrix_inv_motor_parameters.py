import numpy as np
import pytest

from matrix_inv.motor_parameters import (
    estimate_ar_coefficients,
    parabolic_peak_offset,
    refine_f0_hz,
)


def test_symmetric_peak_gives_zero_offset():
    # mag_left == mag_right -> the parabola's vertex is exactly at x=0.
    assert parabolic_peak_offset(1.0, 5.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_offset_leans_toward_the_stronger_neighbor():
    # A slightly stronger right neighbor should pull the interpolated
    # peak toward positive x (matches how a real spectral peak that
    # falls between two bins behaves).
    offset = parabolic_peak_offset(4.0, 5.0, 4.5)
    assert 0.0 < offset < 0.5


def test_offset_is_antisymmetric():
    left_leaning = parabolic_peak_offset(4.5, 5.0, 4.0)
    right_leaning = parabolic_peak_offset(4.0, 5.0, 4.5)
    assert left_leaning == pytest.approx(-right_leaning, abs=1e-9)


def test_degenerate_flat_points_return_zero_offset():
    # No curvature at all (a==0): straight line, not a parabola -- no
    # well-defined vertex, must not divide by zero.
    assert parabolic_peak_offset(3.0, 3.0, 3.0) == 0.0


def test_offset_matches_known_closed_form_formula():
    # Cross-check against the standard quadratic-interpolation formula
    # (Jacobsen/parabolic peak fit): delta = 0.5*(left-right)/(left-2*center+right).
    left, center, right = 2.0, 10.0, 6.0
    expected = 0.5 * (left - right) / (left - 2 * center + right)
    assert parabolic_peak_offset(left, center, right) == pytest.approx(expected, abs=1e-9)


def test_refine_f0_hz_matches_bin_quantized_value_for_a_symmetric_peak():
    spectrum = np.array([0.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.0])
    fs_hz, n_fft = 800.0, 64
    k0 = 3
    assert refine_f0_hz(spectrum, k0, fs_hz, n_fft) == pytest.approx(k0 * fs_hz / n_fft, abs=1e-9)


def test_refine_f0_hz_shifts_toward_a_lopsided_neighbor():
    spectrum = np.array([0.0, 1.0, 8.0, 10.0, 2.0, 1.0, 0.0])
    fs_hz, n_fft = 800.0, 64
    k0 = 3
    bin_hz = fs_hz / n_fft
    refined = refine_f0_hz(spectrum, k0, fs_hz, n_fft)
    # left neighbor (8.0) much stronger than right (2.0) -> peak actually
    # sits a bit below bin 3, refined frequency should be lower.
    assert refined < k0 * bin_hz


def test_refine_f0_hz_falls_back_to_bin_quantized_at_spectrum_edge():
    spectrum = np.array([5.0, 1.0, 1.0, 1.0])
    fs_hz, n_fft = 800.0, 64
    assert refine_f0_hz(spectrum, 0, fs_hz, n_fft) == pytest.approx(0.0, abs=1e-9)
    last = len(spectrum) - 1
    assert refine_f0_hz(spectrum, last, fs_hz, n_fft) == pytest.approx(last * fs_hz / n_fft, abs=1e-9)


def test_ar_coefficients_recover_a_known_pure_tone_generator():
    # AR(2) with a_1=2*cos(w), a_2=-1 exactly generates a pure sinusoid
    # at angular frequency w (the classic "AR(2) is a tunable oscillator"
    # identity) -- fitting AR(2) back to such a signal should recover
    # those two coefficients closely, a strong correctness check beyond
    # just "runs without crashing".
    n = 2000
    w = 2 * np.pi * 0.05  # arbitrary normalized frequency
    t = np.arange(n)
    x = np.sin(w * t)
    coeffs = estimate_ar_coefficients(x, order=2)
    expected = np.array([2 * np.cos(w), -1.0])
    # The biased autocorrelation estimator has a small windowing bias for
    # a finite, non-integer-period segment -- this checks "close to the
    # true generator", not exact recovery.
    np.testing.assert_allclose(coeffs, expected, atol=2e-3)


def test_ar_coefficients_are_zero_for_a_silent_segment():
    assert np.all(estimate_ar_coefficients(np.zeros(100), order=4) == 0.0)


def test_ar_coefficients_shape_matches_requested_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert estimate_ar_coefficients(x, order=4).shape == (4,)
    assert estimate_ar_coefficients(x, order=3).shape == (3,)
