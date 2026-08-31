import numpy as np
import pytest

from lms.reference import DEFAULT_MU, N_TAPS, run_lms


def test_first_sample_output_is_zero_before_any_history():
    # history starts all-zero, so y(0) = dot(weights=0, history=0) = 0
    result = run_lms(np.array([5.0, 1.0, 1.0, 1.0]), mu=0.0)
    assert result.y[0] == 0.0
    assert result.e[0] == 5.0  # e(0) = d(0) - y(0) = x(0) - 0


def test_weights_do_not_change_when_mu_is_zero():
    x = np.random.default_rng(0).uniform(-1, 1, 50)
    result = run_lms(x, mu=0.0)
    assert np.allclose(result.weights_final, np.zeros(N_TAPS))


def test_cycles_per_sample_matches_the_mac_count_estimate():
    # 8 MACs for y(n) + 8 MACs for weight update = 16 core MACs/sample
    # (a real implementation budgets a little overhead on top, ~17-20
    # total -- this checks the MAC count itself, not the overhead).
    result = run_lms(np.zeros(10), mu=0.0)
    assert result.cycles_per_sample == 2 * N_TAPS == 16


def test_ale_converges_and_predicts_a_clean_periodic_signal():
    # No noise at all: with delay=1 (default), the filter should learn to
    # predict a pure sinusoid almost exactly after it converges -- e(n)
    # should shrink close to 0 in the second half of a long-enough run.
    n = 4000
    t = np.arange(n) / 400.0
    x = np.sin(2 * np.pi * 20.0 * t)  # 20 Hz tone at 400 Hz sample rate

    result = run_lms(x, mu=0.01)
    early_error_rms = np.sqrt(np.mean(result.e[:200] ** 2))
    late_error_rms = np.sqrt(np.mean(result.e[-200:] ** 2))
    assert late_error_rms < early_error_rms * 0.1


def test_ale_separates_periodic_signal_from_broadband_noise():
    rng = np.random.default_rng(1)
    n = 8000
    t = np.arange(n) / 400.0
    tone = np.sin(2 * np.pi * 20.0 * t)
    noise = rng.normal(0, 0.5, n)
    x = tone + noise

    result = run_lms(x, mu=0.005)
    late = slice(n - 1000, n)
    # y(n) (the "enhanced" periodic estimate) should correlate strongly
    # with the clean tone in the converged region; a mostly-noise-passthrough
    # filter (e.g. mu too small / not converged) would not.
    correlation = np.corrcoef(result.y[late], tone[late])[0, 1]
    assert correlation > 0.7


def test_default_delay_never_lets_history_see_the_current_sample():
    # Empirically confirms the docstring's claim: with weights=[1,0,...,0]
    # (so y(n) == history[0] exactly), y(n) must equal x(n-1), never x(n)
    # -- the sequential loop's inherent 1-sample decorrelation, independent
    # of the "delay" parameter, which only adds further decorrelation.
    x = np.arange(1, 30, dtype=float)
    result = run_lms(x, mu=0.0, delay=0, weights_init=[1, 0, 0, 0, 0, 0, 0, 0])
    for n in (5, 10, 15):
        assert result.y[n] == pytest.approx(x[n - 1])


def test_increasing_delay_shifts_the_effective_tap_further_into_the_past():
    x = np.arange(1, 30, dtype=float)
    result = run_lms(x, mu=0.0, delay=3, weights_init=[1, 0, 0, 0, 0, 0, 0, 0])
    for n in (10, 15, 20):
        assert result.y[n] == pytest.approx(x[n - 1 - 3])


def test_default_mu_stays_stable_at_the_ingestion_clip_ceiling():
    # Amplitude varies ~100x between a healthy and a severely worn bearing,
    # and a fixed-mu LMS fed raw/unnormalized samples can diverge to inf on
    # the highest-amplitude trials. This project's architecture feeds every
    # module ingestion-normalized samples (clipped to [-1, 1],
    # dataset.signal_params.normalize_ingestion) before anything touches
    # them, which bounds input power to <=1.0 regardless of the raw
    # trial's amplitude -- classic LMS stability (mu < 2/(n_taps*P)) then
    # gives mu < 2/(8*1.0) = 0.25, a 16x margin over DEFAULT_MU
    # (2**-6 = 0.015625). A +-1 square wave is the worst case allowed by
    # the clip (maximum possible power); if DEFAULT_MU is stable here, it
    # is stable for anything the ingestion clip can ever produce.
    square_wave = np.tile([1.0, -1.0], 2000)
    result = run_lms(square_wave, mu=DEFAULT_MU)
    assert np.isfinite(result.weights_final).all()
    assert np.isfinite(result.e).all()
    assert np.max(np.abs(result.weights_final)) < 2.0
