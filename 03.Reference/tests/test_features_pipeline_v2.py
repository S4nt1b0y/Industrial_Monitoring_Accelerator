import numpy as np
import pytest

from features.pipeline import BLOCK_SAMPLES, N_VIBRATION_CHANNELS
from features.pipeline_v2 import AR_ORDER, FeatureVectorV2, extract_features


def _synthetic_current(freq_hz=50.0, n=BLOCK_SAMPLES, amplitude=100.0, noise=0.5):
    from dataset.signal_params import FS_HZ

    t = np.arange(n) / FS_HZ
    rng = np.random.default_rng(0)
    return amplitude * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, noise, n)


def _synthetic_vib_blocks(amplitude=1.0, seed_start=1):
    rng_amplitude = amplitude
    return [
        np.random.default_rng(seed_start + i).normal(0, rng_amplitude, BLOCK_SAMPLES)
        for i in range(N_VIBRATION_CHANNELS)
    ]


def test_extract_features_returns_v1_plus_ar_coefficients():
    result = extract_features(_synthetic_vib_blocks(), _synthetic_current())
    assert isinstance(result, FeatureVectorV2)
    assert len(result.ar_coefficients) == AR_ORDER
    # v1 has 130 values (f0_hz, f0_valido, 128 bins) + 4 AR = 134
    assert len(result.as_tuple()) == 130 + AR_ORDER


def test_v1_part_matches_calling_v1_directly():
    from features.pipeline import extract_features as extract_features_v1

    blocks = _synthetic_vib_blocks()
    cur = _synthetic_current()
    result = extract_features(blocks, cur)
    v1_only = extract_features_v1(blocks, cur)
    # Float equality, not `==`: repeated identical FFT/decimate calls can
    # differ by ~1e-17 (floating-point non-associativity noise, found
    # while writing this test -- not a real bug in either pipeline).
    assert result.v1.f0_hz == pytest.approx(v1_only.f0_hz)
    assert result.v1.f0_valido == v1_only.f0_valido
    np.testing.assert_allclose(result.v1.lowfreq_spectrum, v1_only.lowfreq_spectrum, atol=1e-12)


def test_ar_coefficients_differ_for_a_strongly_periodic_vs_noisy_first_channel():
    cur = _synthetic_current()
    t = np.arange(BLOCK_SAMPLES) / 25_600.0
    periodic_blocks = [10.0 * np.sin(2 * np.pi * 50.0 * t)] + _synthetic_vib_blocks(seed_start=2)[1:]
    noisy_blocks = _synthetic_vib_blocks(amplitude=1.0, seed_start=10)

    periodic_result = extract_features(periodic_blocks, cur)
    noisy_result = extract_features(noisy_blocks, cur)
    assert periodic_result.ar_coefficients != noisy_result.ar_coefficients
