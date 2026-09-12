#!/usr/bin/env python3
"""Send one 64-sample Q1.15 vibration window to the FPGA UART input."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("07.Datasets/processed/motor_measurements_q15.parquet")
DEFAULT_BAUD = 9600
DEFAULT_COUNT = 64
VIBRATION_COLUMNS = (
    "aceleracao_x_mancal_a",
    "aceleracao_y_mancal_a",
    "aceleracao_x_mancal_b",
    "aceleracao_y_mancal_b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read 64 Q1.15 vibration samples from a Parquet file and transmit "
            "them to an FPGA UART RX through a USB-TTL adapter such as CP2102."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--port",
        help="Serial port connected to CP2102, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--offset", type=int, default=0, help="First Parquet row to transmit.")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of samples per channel. Defaults to the RTL window size, 64.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Serial write timeout in seconds.",
    )
    parser.add_argument(
        "--inter-frame-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between the four channel blocks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report the payload without opening the serial port.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List detected serial ports and exit.",
    )
    return parser.parse_args()


def require_pyserial():
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for UART access. Install dependencies with "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from exc
    return serial


def list_serial_ports() -> int:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required to list serial ports. Install dependencies with "
            ".venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected.")
        return 0

    for port in ports:
        print(f"{port.device}\t{port.description}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.offset < 0:
        raise ValueError("--offset must be greater than or equal to zero")
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if args.count != DEFAULT_COUNT:
        raise ValueError("--count must be 64 to match the current RTL window size")
    if args.baud <= 0:
        raise ValueError("--baud must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.inter_frame_delay < 0:
        raise ValueError("--inter-frame-delay must be greater than or equal to zero")
    if not args.dry_run and not args.port:
        raise ValueError("--port is required unless --dry-run or --list-ports is used")


def validate_parquet_schema(path: Path) -> pq.ParquetFile:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    names = set(schema.names)
    missing = sorted(set(VIBRATION_COLUMNS) - names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    channel_types = [schema.field(column).type for column in VIBRATION_COLUMNS]
    if not all(pa.types.is_int16(channel_type) for channel_type in channel_types):
        raise ValueError(
            f"{path} must contain Q1.15 int16 vibration columns; got {channel_types}"
        )

    return parquet_file


def read_window(path: Path, offset: int, count: int) -> np.ndarray:
    parquet_file = validate_parquet_schema(path)
    rows = parquet_file.metadata.num_rows
    if offset + count > rows:
        raise ValueError(
            f"requested rows [{offset}, {offset + count}) but {path} has only {rows} rows"
        )

    remaining_skip = offset
    collected = []
    collected_rows = 0
    batch_size = max(DEFAULT_COUNT, 4096)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(VIBRATION_COLUMNS),
    ):
        if remaining_skip >= batch.num_rows:
            remaining_skip -= batch.num_rows
            continue

        take = min(count - collected_rows, batch.num_rows - remaining_skip)
        if take <= 0:
            break

        sliced = batch.slice(remaining_skip, take)
        channels = [
            sliced.column(column).to_numpy(zero_copy_only=False)
            for column in VIBRATION_COLUMNS
        ]
        collected.append(np.column_stack(channels).astype(np.int16, copy=False))
        collected_rows += take
        remaining_skip = 0

        if collected_rows == count:
            break

    if not collected:
        raise ValueError(f"could not read any samples from {path}")

    samples = np.vstack(collected)
    if samples.shape[0] != count:
        raise ValueError(f"read {samples.shape[0]} samples from {path}, expected {count}")
    return samples


def build_uart_payload(samples: np.ndarray) -> bytes:
    if samples.ndim != 2 or samples.shape[1] != len(VIBRATION_COLUMNS):
        raise ValueError(
            f"samples must have shape (N, {len(VIBRATION_COLUMNS)}), got {samples.shape}"
        )
    if samples.shape[0] != DEFAULT_COUNT:
        raise ValueError(f"samples must contain exactly {DEFAULT_COUNT} rows")

    payload = bytearray()
    for channel_index in range(len(VIBRATION_COLUMNS)):
        channel = np.asarray(samples[:, channel_index], dtype=">i2")
        payload.extend(channel.tobytes())
    return bytes(payload)


def print_payload_summary(
    dataset: Path,
    offset: int,
    samples: np.ndarray,
    payload: bytes,
) -> None:
    print(f"Dataset: {dataset}")
    print(f"Offset: {offset}")
    print(f"Samples per channel: {samples.shape[0]}")
    print(f"Channels: {len(VIBRATION_COLUMNS)}")
    print(f"Payload bytes: {len(payload)}")
    print(f"First 32 bytes: {payload[:32].hex(' ')}")


def send_payload(
    payload: bytes,
    port: str,
    baud: int,
    timeout: float,
    inter_frame_delay: float,
) -> None:
    serial = require_pyserial()
    with serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=timeout,
    ) as uart:
        if inter_frame_delay == 0:
            written = uart.write(payload)
            uart.flush()
            if written != len(payload):
                raise RuntimeError(f"wrote {written} bytes, expected {len(payload)}")
            return

        frame_size = DEFAULT_COUNT * 2
        total_written = 0
        for start in range(0, len(payload), frame_size):
            frame = payload[start : start + frame_size]
            total_written += uart.write(frame)
            uart.flush()
            if start + frame_size < len(payload):
                time.sleep(inter_frame_delay)
        if total_written != len(payload):
            raise RuntimeError(f"wrote {total_written} bytes, expected {len(payload)}")


def main() -> int:
    args = parse_args()
    try:
        if args.list_ports:
            return list_serial_ports()

        validate_args(args)
        samples = read_window(args.dataset, args.offset, args.count)
        payload = build_uart_payload(samples)
        print_payload_summary(args.dataset, args.offset, samples, payload)

        if args.dry_run:
            print("Dry run: UART port was not opened.")
            return 0

        send_payload(payload, args.port, args.baud, args.timeout, args.inter_frame_delay)
        print(f"Sent {len(payload)} bytes to {args.port} at {args.baud} baud.")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
