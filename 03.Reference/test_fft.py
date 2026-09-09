#!/usr/bin/env python3
"""Unit tests for fixed-point FFT helpers."""

from __future__ import annotations

import unittest

import numpy as np

from fft import fft_magnitude_q, fft_magnitude_q_batch


class TestFFT(unittest.TestCase):
    def test_fft_magnitude_q17_is_saturated_to_8_bit_feature_range(self) -> None:
        signal = np.full(64, 127, dtype=np.int8)
        magnitude = fft_magnitude_q(signal, data_width=8)

        self.assertEqual(magnitude.shape, (64,))
        self.assertEqual(magnitude.dtype, np.int32)
        self.assertTrue(np.all(magnitude >= 0))
        self.assertTrue(np.all(magnitude <= 127))

    def test_fft_magnitude_q15_keeps_existing_16_bit_feature_range(self) -> None:
        signal = np.full(64, 1024, dtype=np.int16)
        magnitude = fft_magnitude_q(signal, data_width=16)

        self.assertEqual(magnitude.shape, (64,))
        self.assertEqual(magnitude.dtype, np.int32)
        self.assertTrue(np.all(magnitude >= 0))
        self.assertTrue(np.all(magnitude <= 32767))

    def test_rejects_out_of_range_q17_samples(self) -> None:
        signal = np.full(64, 128, dtype=np.int16)

        with self.assertRaises(ValueError):
            fft_magnitude_q(signal, data_width=8)

    def test_batch_matches_scalar_fft_magnitude(self) -> None:
        signals = np.asarray(
            [
                np.arange(64, dtype=np.int16) % 16,
                np.full(64, 32, dtype=np.int16),
            ]
        )

        batch = fft_magnitude_q_batch(signals, data_width=8)
        scalar = np.vstack([fft_magnitude_q(signal, data_width=8) for signal in signals])

        np.testing.assert_array_equal(batch, scalar)


if __name__ == "__main__":
    unittest.main()
