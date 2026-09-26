#!/usr/bin/env python3
"""Send Q1.15 vibration windows continuously or as a single burst to FPGA UART input."""

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
            "Read Q1.15 vibration samples from a Parquet file and transmit "
            "them continuously or once to an FPGA UART RX."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--port",
        help="Serial port connected to CP2102, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--offset", type=int, default=0, help="Initial Parquet row to transmit."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of samples per channel. Defaults to 64.",
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
        help="Delay in seconds between the four channel blocks within a frame.",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=0.1,
        help="Delay in seconds between consecutive windows in loop mode.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously read and send windows in a 'while True' loop.",
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
            "pyserial is required to list serial ports."
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
        raise ValueError("--count must be 64 to match current RTL window size")
    if args.baud <= 0:
        raise ValueError("--baud must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.inter_frame_delay < 0:
        raise ValueError("--inter-frame-delay must be greater than or equal to zero")
    if args.loop_delay < 0:
        raise ValueError("--loop-delay must be greater than or equal to zero")
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


def load_dataset_matrix(path: Path) -> np.ndarray:
    """Loads the entire Parquet dataset into memory as a 2D int16 NumPy array."""
    validate_parquet_schema(path)
    table = pq.read_table(path, columns=list(VIBRATION_COLUMNS))
    channels = [
        table.column(col).to_numpy(zero_copy_only=False)
        for col in VIBRATION_COLUMNS
    ]
    return np.column_stack(channels).astype(np.int16, copy=False)


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


def send_payload_uart(
    uart,
    payload: bytes,
    inter_frame_delay: float,
) -> None:
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
        
        print(f"Loading dataset {args.dataset} into memory...")
        data_matrix = load_dataset_matrix(args.dataset)
        total_rows = data_matrix.shape[0]
        print(f"Dataset loaded successfully ({total_rows} total rows).")

        current_offset = args.offset
        if current_offset >= total_rows:
            raise ValueError(f"Initial offset {current_offset} exceeds total rows ({total_rows})")

        serial = None
        if not args.dry_run:
            pyserial = require_pyserial()
            serial = pyserial.Serial(
                port=args.port,
                baudrate=args.baud,
                bytesize=8,
                parity="N",
                stopbits=1,
                timeout=args.timeout,
            )

        print(f"Starting transmission (Loop Mode: {args.loop})... Press Ctrl+C to stop.")

        tx_count = 0
        try:
            while True:
                # Se atingir o fim dos dados, reinicia do offset inicial (ou de 0)
                if current_offset + args.count > total_rows:
                    print("\nReached end of dataset. Wrapping around to row 0.")
                    current_offset = 0

                samples = data_matrix[current_offset : current_offset + args.count]
                payload = build_uart_payload(samples)

                if serial:
                    send_payload_uart(serial, payload, args.inter_frame_delay)
                
                tx_count += 1
                sys.stdout.write(
                    f"\r[TX #{tx_count}] Offset: {current_offset:<8} | Sent {len(payload)} bytes"
                )
                sys.stdout.flush()

                if not args.loop:
                    print()
                    break

                current_offset += args.count
                if args.loop_delay > 0:
                    time.sleep(args.loop_delay)

        except KeyboardInterrupt:
            print("\nTransmission stopped by user.")
        finally:
            if serial and serial.is_open:
                serial.close()

        return 0

    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())