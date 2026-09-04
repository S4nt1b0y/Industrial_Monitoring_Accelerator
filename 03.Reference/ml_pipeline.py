#!/usr/bin/env python3
"""Top-level reference pipeline that feeds fixed-point samples into MLClassifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fft import fft_magnitude_q, fft_magnitude_q_batch
from lms import LMSHardwareModel
from mdc import mdc_features_from_magnitude, mdc_features_from_magnitude_batch
from ml_classifier import MLClassifier, SUPPORTED_DATA_WIDTHS


CHANNEL_COUNT = 4
WINDOW_SIZE = 64
FIRST_FFT_BIN = 0
LAST_FFT_BIN = WINDOW_SIZE // 2
FFT_BIN_COUNT = LAST_FFT_BIN - FIRST_FFT_BIN + 1
FFT_FEATURE_COUNT = CHANNEL_COUNT * FFT_BIN_COUNT
MDC_FEATURES_PER_CHANNEL = ("f0", "valid")
MDC_FEATURE_COUNT = CHANNEL_COUNT * len(MDC_FEATURES_PER_CHANNEL)
DEFAULT_OUTPUT_ROOT = Path("03.Reference/artifacts/top_classifier")
DEFAULT_FS_HZ = 6400
DEFAULT_MIN_K = 2
DEFAULT_LMS_DELAY = 1
DEFAULT_LMS_TAPS = 8
DEFAULT_LMS_MU = 0.01


class MLPipeline:
    """Fixed FFT + MDC pipeline connected to MLClassifier, with optional LMS compatibility."""

    def __init__(
        self,
        data_width: int = 8,
        *,
        lms: bool = False,
        mdc: bool = True,
        fs_hz: int = DEFAULT_FS_HZ,
        min_k: int = DEFAULT_MIN_K,
        lms_delay: int = DEFAULT_LMS_DELAY,
        lms_taps: int = DEFAULT_LMS_TAPS,
        lms_mu: float = DEFAULT_LMS_MU,
        max_depth: int = 5,
        min_samples_leaf: int = 16,
        seed: int = 42,
    ) -> None:
        if data_width not in SUPPORTED_DATA_WIDTHS:
            raise ValueError(f"data_width must be one of {SUPPORTED_DATA_WIDTHS}, got {data_width}")
        if fs_hz <= 0:
            raise ValueError(f"fs_hz must be positive, got {fs_hz}")
        if min_k < 0:
            raise ValueError(f"min_k must be non-negative, got {min_k}")
        if lms and (lms_delay <= 0 or lms_delay >= WINDOW_SIZE):
            raise ValueError(f"lms_delay must be between 1 and {WINDOW_SIZE - 1}, got {lms_delay}")

        self.data_width = int(data_width)
        self.q_format = f"q1_{self.data_width - 1}"
        self.lms = bool(lms)
        self.mdc = bool(mdc)
        self.fs_hz = int(fs_hz)
        self.min_k = int(min_k)
        self.lms_delay = int(lms_delay)
        self.lms_taps = int(lms_taps)
        self.lms_mu = float(lms_mu)
        self.window_size = WINDOW_SIZE
        self.scale = float(2 ** (self.data_width - 1))
        self.min_sample_value = -(2 ** (self.data_width - 1))
        self.max_sample_value = (2 ** (self.data_width - 1)) - 1
        self.max_feature_value = self.max_sample_value
        self.n_features = FFT_FEATURE_COUNT + (MDC_FEATURE_COUNT if self.mdc else 0)
        self.sample_buffers: list[list[int]] = [[] for _ in range(CHANNEL_COUNT)]
        self.ml_classifier = MLClassifier(
            n_features=self.n_features,
            data_width=self.data_width,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
        )

    def train(
        self,
        ch0: np.ndarray,
        ch1: np.ndarray,
        ch2: np.ndarray,
        ch3: np.ndarray,
        labels: np.ndarray,
        *,
        test_size: float = 0.15,
        val_size: float = 0.15,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        channels = self._validate_channels((ch0, ch1, ch2, ch3))
        labels = np.asarray(labels)
        window_count = channels[0].shape[0] // self.window_size
        if labels.ndim != 1:
            raise ValueError(f"labels must be a 1-D array, got shape {labels.shape}")
        if labels.shape[0] != window_count:
            raise ValueError(
                f"labels must have one class per {self.window_size}-sample window: "
                f"expected {window_count}, got {labels.shape[0]}"
            )

        features = self.channels_to_features(channels)
        metrics = self.ml_classifier.train(
            features,
            labels,
            test_size=test_size,
            val_size=val_size,
            cv_folds=cv_folds,
        )
        metrics["pipeline"] = self.config()
        return metrics

    def classifier(
        self,
        sample_ch0: int,
        sample_ch1: int,
        sample_ch2: int,
        sample_ch3: int,
    ) -> tuple[bool, int | None]:
        samples = np.asarray([sample_ch0, sample_ch1, sample_ch2, sample_ch3])
        self._validate_sample_values(samples)
        for channel_index, sample in enumerate(samples):
            self.sample_buffers[channel_index].append(int(sample))

        if len(self.sample_buffers[0]) < self.window_size:
            return False, None

        window = np.column_stack(
            [
                np.asarray(buffer[: self.window_size], dtype=np.int32)
                for buffer in self.sample_buffers
            ]
        )
        self.sample_buffers = [[] for _ in range(CHANNEL_COUNT)]
        features = self.window_to_features(window).reshape(1, self.n_features)
        class_id = int(self.ml_classifier.classify(features)[0])
        return True, class_id

    def classify(
        self,
        sample_ch0: int,
        sample_ch1: int,
        sample_ch2: int,
        sample_ch3: int,
    ) -> tuple[bool, int | None]:
        return self.classifier(sample_ch0, sample_ch1, sample_ch2, sample_ch3)

    def save_artifacts(self, output_dir: Path | str) -> None:
        output_path = Path(output_dir)
        self.ml_classifier.save_artifacts(output_path)
        (output_path / "pipeline_config.json").write_text(
            json.dumps(self.config(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def channels_to_features(self, channels: tuple[np.ndarray, ...]) -> np.ndarray:
        channels = self._validate_channels(channels)
        sample_count = channels[0].shape[0]
        window_count = sample_count // self.window_size
        if window_count == 0:
            raise ValueError(f"at least {self.window_size} samples per channel are required")

        trimmed_count = window_count * self.window_size
        stacked = np.column_stack([channel[:trimmed_count] for channel in channels])
        windows = stacked.reshape(window_count, self.window_size, CHANNEL_COUNT)
        if not self.lms:
            return self._windows_to_features_batch(windows)

        features = np.empty((window_count, self.n_features), dtype=np.int32)
        for window_index, window in enumerate(windows):
            features[window_index] = self.window_to_features(window)
        return features

    def _windows_to_features_batch(self, windows: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        features = np.empty((windows.shape[0], self.n_features), dtype=np.int32)
        for start in range(0, windows.shape[0], chunk_size):
            stop = min(start + chunk_size, windows.shape[0])
            chunk = windows[start:stop]
            fft_features = []
            mdc_features = []
            for channel_index in range(CHANNEL_COUNT):
                magnitude = fft_magnitude_q_batch(
                    chunk[:, :, channel_index],
                    self.data_width,
                    window_size=self.window_size,
                )
                unique_bins = magnitude[:, FIRST_FFT_BIN : LAST_FFT_BIN + 1].astype(np.int32)
                fft_features.append(unique_bins)
                if self.mdc:
                    mdc_features.append(
                        mdc_features_from_magnitude_batch(
                            unique_bins,
                            self.data_width,
                            fs_hz=self.fs_hz,
                            min_k=self.min_k,
                            n_fft=self.window_size,
                        )
                    )
            features[start:stop] = np.concatenate([*fft_features, *mdc_features], axis=1)
        return features

    def window_to_features(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window)
        if window.shape != (self.window_size, CHANNEL_COUNT):
            raise ValueError(
                f"window must have shape ({self.window_size}, {CHANNEL_COUNT}), got {window.shape}"
            )
        self._validate_sample_values(window)

        fft_features = []
        mdc_features = []
        for channel_index in range(CHANNEL_COUNT):
            channel = window[:, channel_index]
            processed_channel = self._apply_lms(channel) if self.lms else channel
            magnitude = self._fft_magnitude_features(processed_channel, input_is_float=self.lms)
            fft_features.append(magnitude)

            if self.mdc:
                mdc_features.append(
                    mdc_features_from_magnitude(
                        magnitude,
                        self.data_width,
                        fs_hz=self.fs_hz,
                        min_k=self.min_k,
                        n_fft=self.window_size,
                    )
                )

        return np.concatenate([*fft_features, *mdc_features]).astype(np.int32)

    def config(self) -> dict[str, Any]:
        return {
            "data_width": self.data_width,
            "q_format": self.q_format,
            "window_size": self.window_size,
            "lms": self.lms,
            "mdc": self.mdc,
            "n_features": self.n_features,
            "sample_range": {
                "min": self.min_sample_value,
                "max": self.max_sample_value,
            },
            "feature_range": {
                "min": 0,
                "max": self.max_feature_value,
            },
            "fft": {
                "bins": {
                    "first": FIRST_FFT_BIN,
                    "last": LAST_FFT_BIN,
                    "count_per_channel": FFT_BIN_COUNT,
                },
                "feature_count": FFT_FEATURE_COUNT,
            },
            "lms_config": {
                "enabled": self.lms,
                "delay": self.lms_delay if self.lms else None,
                "taps": self.lms_taps if self.lms else None,
                "mu": self.lms_mu if self.lms else None,
            },
            "mdc_config": {
                "enabled": self.mdc,
                "features_per_channel": list(MDC_FEATURES_PER_CHANNEL) if self.mdc else [],
                "feature_count": MDC_FEATURE_COUNT if self.mdc else 0,
                "fs_hz": self.fs_hz if self.mdc else None,
                "min_k": self.min_k if self.mdc else None,
            },
            "feature_layout": self.feature_layout(),
        }

    def feature_layout(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        feature_index = 0
        for channel_index in range(CHANNEL_COUNT):
            for bin_index in range(FIRST_FFT_BIN, LAST_FFT_BIN + 1):
                rows.append(
                    {
                        "feature_index": feature_index,
                        "channel": channel_index,
                        "source": "fft_magnitude",
                        "bin": bin_index,
                    }
                )
                feature_index += 1

        if self.mdc:
            for channel_index in range(CHANNEL_COUNT):
                for feature_name in MDC_FEATURES_PER_CHANNEL:
                    rows.append(
                        {
                            "feature_index": feature_index,
                            "channel": channel_index,
                            "source": "mdc",
                            "name": feature_name,
                        }
                    )
                    feature_index += 1
        return rows

    def _apply_lms(self, signal: np.ndarray) -> np.ndarray:
        signal_float = self._to_float(signal)
        filtered = np.empty_like(signal_float)
        lms_model = LMSHardwareModel(num_taps=self.lms_taps, mu=self.lms_mu)

        for index, desired in enumerate(signal_float):
            reference = signal_float[index - self.lms_delay] if index >= self.lms_delay else 0.0
            result = lms_model.process_sample(x_new=float(reference), d_new=float(desired))
            filtered[index] = result["y"]
        return filtered

    def _fft_magnitude_features(
        self,
        signal: np.ndarray,
        *,
        input_is_float: bool = False,
    ) -> np.ndarray:
        magnitude = fft_magnitude_q(
            signal,
            self.data_width,
            window_size=self.window_size,
            input_is_float=input_is_float,
        )
        return magnitude[FIRST_FFT_BIN : LAST_FFT_BIN + 1].astype(np.int32)

    def _to_float(self, signal: np.ndarray) -> np.ndarray:
        return signal.astype(np.float64) / self.scale

    def _validate_channels(self, channels: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        if len(channels) != CHANNEL_COUNT:
            raise ValueError(f"expected {CHANNEL_COUNT} channels, got {len(channels)}")

        arrays = tuple(np.asarray(channel) for channel in channels)
        lengths = [array.shape[0] for array in arrays if array.ndim == 1]
        if len(lengths) != CHANNEL_COUNT:
            shapes = [array.shape for array in arrays]
            raise ValueError(f"each channel must be a 1-D array, got shapes {shapes}")
        if len(set(lengths)) != 1:
            raise ValueError(f"all channels must have the same length, got {lengths}")

        for array in arrays:
            self._validate_sample_values(array)
        return tuple(array.astype(np.int32, copy=False) for array in arrays)

    def _validate_sample_values(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"samples must contain integer fixed-point values, got {values.dtype}")
        if values.size == 0:
            return

        min_value = int(np.min(values))
        max_value = int(np.max(values))
        if min_value < self.min_sample_value or max_value > self.max_sample_value:
            raise ValueError(
                "samples out of range for "
                f"{self.q_format}: expected "
                f"{self.min_sample_value}..{self.max_sample_value}, "
                f"got {min_value}..{max_value}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLPipeline from four preloaded .npy fixed-point channels."
    )
    parser.add_argument("--ch0", type=Path, required=True)
    parser.add_argument("--ch1", type=Path, required=True)
    parser.add_argument("--ch2", type=Path, required=True)
    parser.add_argument("--ch3", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-width", type=int, choices=SUPPORTED_DATA_WIDTHS, default=8)
    parser.set_defaults(lms=False, mdc=True)
    parser.add_argument(
        "--enable-lms",
        dest="lms",
        action="store_true",
        help="Compatibility option: enable the legacy LMS stage before FFT.",
    )
    parser.add_argument(
        "--disable-lms",
        dest="lms",
        action="store_false",
        help="Compatibility option: LMS is already disabled by default.",
    )
    parser.add_argument(
        "--mdc",
        dest="mdc",
        action="store_true",
        help="Compatibility option: MDC is already enabled by default.",
    )
    parser.add_argument(
        "--disable-mdc",
        dest="mdc",
        action="store_false",
        help="Compatibility option: disable MDC features.",
    )
    parser.add_argument("--fs-hz", type=int, default=DEFAULT_FS_HZ)
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--lms-delay", type=int, default=DEFAULT_LMS_DELAY)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def train_from_args(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = MLPipeline(
        data_width=args.data_width,
        lms=args.lms,
        mdc=args.mdc,
        fs_hz=args.fs_hz,
        min_k=args.min_k,
        lms_delay=args.lms_delay,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    metrics = pipeline.train(
        np.load(args.ch0),
        np.load(args.ch1),
        np.load(args.ch2),
        np.load(args.ch3),
        np.load(args.labels),
        test_size=args.test_size,
        val_size=args.val_size,
        cv_folds=args.cv_folds,
    )
    pipeline.save_artifacts(args.output_dir)

    print(
        f"Trained MLPipeline with data_width={pipeline.data_width}, "
        f"lms={pipeline.lms}, mdc={pipeline.mdc}, n_features={pipeline.n_features}"
    )
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Wrote artifacts to {args.output_dir}")
    return metrics


def main() -> None:
    train_from_args(parse_args())


if __name__ == "__main__":
    main()
