#!/usr/bin/env python3
"""Unit tests for MDC feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from mdc import mdc_features_from_magnitude, mdc_features_from_magnitude_batch


class TestMDC(unittest.TestCase):
    def test_saturates_f0_to_q17_feature_range(self) -> None:
        magnitude = np.zeros(33, dtype=np.int32)
        magnitude[[8, 16, 24]] = [90, 80, 70]

        features = mdc_features_from_magnitude(
            magnitude,
            data_width=8,
            fs_hz=6400,
            min_k=2,
            n_fft=64,
        )

        np.testing.assert_array_equal(features, np.asarray([127, 1], dtype=np.int32))

    def test_rejects_zero_peak_bin(self) -> None:
        magnitude = np.zeros(33, dtype=np.int32)
        magnitude[[0, 8, 16]] = [90, 80, 70]

        features = mdc_features_from_magnitude(
            magnitude,
            data_width=8,
            fs_hz=6400,
            min_k=2,
            n_fft=64,
        )

        np.testing.assert_array_equal(features, np.asarray([0, 0], dtype=np.int32))

    def test_invalid_when_mdc_is_below_min_k(self) -> None:
        magnitude = np.zeros(33, dtype=np.int32)
        magnitude[[12, 17, 31]] = [90, 80, 70]

        features = mdc_features_from_magnitude(
            magnitude,
            data_width=16,
            fs_hz=6400,
            min_k=2,
            n_fft=64,
        )

        np.testing.assert_array_equal(features, np.asarray([0, 0], dtype=np.int32))

    def test_batch_matches_scalar_mdc_features(self) -> None:
        magnitudes = np.zeros((3, 33), dtype=np.int32)
        magnitudes[0, [8, 16, 24]] = [90, 80, 70]
        magnitudes[1, [0, 8, 16]] = [90, 80, 70]
        magnitudes[2, [12, 17, 31]] = [90, 80, 70]

        batch = mdc_features_from_magnitude_batch(
            magnitudes,
            data_width=8,
            fs_hz=6400,
            min_k=2,
            n_fft=64,
        )
        scalar = np.vstack(
            [
                mdc_features_from_magnitude(
                    magnitude,
                    data_width=8,
                    fs_hz=6400,
                    min_k=2,
                    n_fft=64,
                )
                for magnitude in magnitudes
            ]
        )

        np.testing.assert_array_equal(batch, scalar)


if __name__ == "__main__":
    unittest.main()
