import numpy as np
import pytest

from features.pipeline import BLOCK_SAMPLES, N_VIBRATION_CHANNELS, FeatureVector, extract_features
from dataset.signal_params import FS_HZ


def _synthetic_current(freq_hz=50.0, n=BLOCK_SAMPLES, fs_hz=FS_HZ, amplitude=100.0, noise=0.5):
    t = np.arange(n) / fs_hz
    rng = np.random.default_rng(0)
    return amplitude * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, noise, n)


def _synthetic_vibration(n=BLOCK_SAMPLES, amplitude=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, n)


def _synthetic_vib_blocks(amplitude=1.0, seed_start=1):
    return [_synthetic_vibration(amplitude=amplitude, seed=seed_start + i) for i in range(N_VIBRATION_CHANNELS)]


def test_extract_features_returns_a_feature_vector_with_all_fields():
    result = extract_features(_synthetic_vib_blocks(), _synthetic_current())
    assert isinstance(result, FeatureVector)
    assert len(result.as_tuple()) == 130  # f0_hz, f0_valido, 4 channels x 32 bins


def test_rejects_wrong_number_of_vibration_channels():
    with pytest.raises(ValueError):
        extract_features(_synthetic_vib_blocks()[:2], _synthetic_current())


def test_rejects_wrong_block_size():
    with pytest.raises(ValueError):
        extract_features([np.zeros(100)] * N_VIBRATION_CHANNELS, np.zeros(100))


def test_f0_recovers_a_clean_dominant_tone_in_current():
    # Strong, clean 50 Hz tone (current's real-world profile) --
    # dominance fallback should read it off directly and mark it valid.
    cur = _synthetic_current(freq_hz=50.0, amplitude=100.0, noise=0.1)
    result = extract_features(_synthetic_vib_blocks(), cur)
    assert result.f0_valido
    assert result.f0_hz == pytest.approx(50.0, abs=15.0)  # Delta_f=12.5Hz at 32x decimation


def test_lowfreq_spectrum_bin_is_higher_when_a_matching_1x_tone_is_present():
    # decimated fs = FS_HZ/VIBRATION_DECIM_FACTOR = 400Hz, bin 8 -> 8*400/64 = 50Hz --
    # inject a clean 50Hz tone (native domain) into the first channel (x_A)
    # and check it survives decimation into the corresponding bin.
    cur = _synthetic_current()
    t = np.arange(BLOCK_SAMPLES) / FS_HZ
    loud_x_a = 10.0 * np.sin(2 * np.pi * 50.0 * t)

    quiet_blocks = _synthetic_vib_blocks(amplitude=0.1)
    loud_blocks = [loud_x_a] + _synthetic_vib_blocks(amplitude=0.1, seed_start=2)[1:]

    quiet_result = extract_features(quiet_blocks, cur)
    loud_result = extract_features(loud_blocks, cur)
    bin_index = 8 - 1  # LOWFREQ_BINS starts at bin 1, so bin 8 is tuple index 7 (channel x_A first)
    assert loud_result.lowfreq_spectrum[bin_index] > quiet_result.lowfreq_spectrum[bin_index] * 10


def test_lowfreq_spectrum_covers_every_channel_in_order():
    # A tone only in the 3rd channel (x_B) should show up only in that
    # channel's 32-bin slice of the concatenated spectrum, not the others.
    cur = _synthetic_current()
    t = np.arange(BLOCK_SAMPLES) / FS_HZ
    loud_x_b = 10.0 * np.sin(2 * np.pi * 50.0 * t)

    quiet_blocks = _synthetic_vib_blocks(amplitude=0.1)
    blocks = list(_synthetic_vib_blocks(amplitude=0.1, seed_start=10))
    blocks[2] = loud_x_b

    quiet_result = extract_features(quiet_blocks, cur)
    loud_result = extract_features(blocks, cur)

    bin_index_in_x_b = 2 * 32 + (8 - 1)  # 3rd channel's slice, bin 8
    assert loud_result.lowfreq_spectrum[bin_index_in_x_b] > quiet_result.lowfreq_spectrum[bin_index_in_x_b] * 10
    # other channels' slices should be unaffected (still quiet-level)
    bin_index_in_x_a = 0 * 32 + (8 - 1)
    assert loud_result.lowfreq_spectrum[bin_index_in_x_a] < quiet_result.lowfreq_spectrum[bin_index_in_x_b] * 0.5
