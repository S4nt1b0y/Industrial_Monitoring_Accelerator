#!/usr/bin/env python3
"""Unit tests for uart_tx_parquet.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uart_tx_parquet import (
    DEFAULT_COUNT,
    VIBRATION_COLUMNS,
    build_uart_payload,
    read_window,
)


def write_test_parquet(path: Path, dtype: np.dtype) -> None:
    arrays = {}
    for channel_index, column in enumerate(VIBRATION_COLUMNS):
        values = np.arange(DEFAULT_COUNT + 4, dtype=np.int16) + channel_index * 100
        arrays[column] = pa.array(values.astype(dtype))
    pq.write_table(pa.table(arrays), path)


class TestUartTxParquet(unittest.TestCase):
    def test_builds_big_endian_channel_ordered_payload(self) -> None:
        samples = np.zeros((DEFAULT_COUNT, len(VIBRATION_COLUMNS)), dtype=np.int16)
        for sample_index in range(DEFAULT_COUNT):
            samples[sample_index] = [
                sample_index,
                0x0100 + sample_index,
                -sample_index,
                -0x0100 - sample_index,
            ]

        payload = build_uart_payload(samples)

        self.assertEqual(len(payload), 4 * DEFAULT_COUNT * 2)
        self.assertEqual(payload[:8], bytes.fromhex("00 00 00 01 00 02 00 03"))
        self.assertEqual(
            payload[DEFAULT_COUNT * 2 : DEFAULT_COUNT * 2 + 8],
            bytes.fromhex("01 00 01 01 01 02 01 03"),
        )
        self.assertEqual(
            payload[2 * DEFAULT_COUNT * 2 : 2 * DEFAULT_COUNT * 2 + 8],
            bytes.fromhex("00 00 ff ff ff fe ff fd"),
        )

    def test_reads_64_sample_window_from_q15_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q15.parquet"
            write_test_parquet(path, np.dtype(np.int16))

            samples = read_window(path, offset=2, count=DEFAULT_COUNT)

            self.assertEqual(samples.shape, (DEFAULT_COUNT, len(VIBRATION_COLUMNS)))
            self.assertEqual(samples.dtype, np.int16)
            self.assertEqual(samples[0].tolist(), [2, 102, 202, 302])
            self.assertEqual(samples[-1].tolist(), [65, 165, 265, 365])

    def test_rejects_q17_int8_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q17.parquet"
            write_test_parquet(path, np.dtype(np.int8))

            with self.assertRaisesRegex(ValueError, "Q1.15 int16"):
                read_window(path, offset=0, count=DEFAULT_COUNT)


if __name__ == "__main__":
    unittest.main()
