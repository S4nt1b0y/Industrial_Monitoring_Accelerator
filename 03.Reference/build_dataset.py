#!/usr/bin/env python3
"""Build a Q1.15 consolidated motor measurements dataset from paired TDMS and MAT files."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nptdms import TdmsFile
from scipy.io import loadmat


DEFAULT_INPUT_DIR = Path("07.Datasets")
DEFAULT_OUTPUT = Path("07.Datasets/processed/motor_measurements_q15.parquet")
DEFAULT_REPORT = Path("07.Datasets/processed/dataset_report.csv")
DEFAULT_SCALE_REPORT = Path("07.Datasets/processed/dataset_scale_report.csv")
DEFAULT_PREVIEW = Path("07.Datasets/processed/dataset_preview.csv")

LABELS = {
    "Normal": "operacao_normal",
    "Unbalance": "desbalanceamento",
    "Misalign": "desalinhamento",
    "BPFO": "desgaste_rolamento",
    "BPFI": "desgaste_rolamento",
}

VIBRATION_COLUMNS = [
    "aceleracao_x_mancal_a",
    "aceleracao_y_mancal_a",
    "aceleracao_x_mancal_b",
    "aceleracao_y_mancal_b",
]

TDMS_COLUMN_BY_CHANNEL = {
    ("Temperature", "Mod1", "ai0"): "temperatura_mancal_a",
    ("Temperature", "Mod1", "ai1"): "temperatura_mancal_b",
    ("Current", "Mod2", "ai0"): "corrente_fase_u",
    ("Current", "Mod2", "ai2"): "corrente_fase_v",
    ("Current", "Mod2", "ai3"): "corrente_fase_w",
}

TDMS_MEASUREMENT_COLUMNS = [
    "temperatura_mancal_a",
    "temperatura_mancal_b",
    "corrente_fase_u",
    "corrente_fase_v",
    "corrente_fase_w",
]

MEASUREMENT_COLUMNS = VIBRATION_COLUMNS + TDMS_MEASUREMENT_COLUMNS
TDMS_VALIDITY_COLUMNS = [f"{column}_valida" for column in TDMS_MEASUREMENT_COLUMNS]
Q15_SCALE = 32768.0


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    label: str
    fault_detail: str


@dataclass
class MatSignal:
    values: np.ndarray
    start_value: float
    increment: float

    @property
    def sample_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_s(self) -> float:
        return self.sample_count * self.increment


@dataclass
class TdmsChannel:
    column_name: str
    channel_type: str
    unit: str
    values: np.ndarray
    start_offset: float
    increment: float

    @property
    def sample_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_s(self) -> float:
        return self.sample_count * self.increment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate paired TDMS current/temperature data and MAT vibration "
            "data into one time-aligned wide Parquet dataset."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scale-report", type=Path, default=DEFAULT_SCALE_REPORT)
    parser.add_argument("--preview-csv", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=1000,
        help="Number of first consolidated rows to save in the preview CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows per Parquet write chunk.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process only the first N source pairs; useful for smoke tests.",
    )
    return parser.parse_args()


def unwrap(value: Any) -> Any:
    """Unwrap scipy MAT structs/cells until a useful Python object remains."""
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def get_field(obj: Any, name: str) -> Any:
    obj = unwrap(obj)
    if hasattr(obj, name):
        return unwrap(getattr(obj, name))
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return unwrap(obj[name])
    raise KeyError(f"MAT field not found: {name}")


def load_mat_signal(path: Path) -> MatSignal:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    signal = mat["Signal"]
    x_values = get_field(signal, "x_values")
    y_values = get_field(signal, "y_values")

    values = np.asarray(get_field(y_values, "values"), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D vibration values, got {values.shape}")

    start_value = float(np.asarray(get_field(x_values, "start_value")).item())
    increment = float(np.asarray(get_field(x_values, "increment")).item())
    number_of_values = int(np.asarray(get_field(x_values, "number_of_values")).item())
    if values.shape[0] != number_of_values:
        raise ValueError(
            f"{path.name}: x_values.number_of_values={number_of_values} "
            f"but y_values.values has {values.shape[0]} rows"
        )

    return MatSignal(values=values, start_value=start_value, increment=increment)


def parse_source_metadata(stem: str) -> SourceMetadata:
    match = re.fullmatch(r"(?P<load>\d+)Nm_(?P<fault>.+)", stem)
    if not match:
        raise ValueError(f"Cannot parse source name: {stem}")

    fault = match.group("fault")
    for token, label in LABELS.items():
        if fault.startswith(token):
            if token == "Normal":
                detail = "normal"
            elif token == "Unbalance":
                detail = "unbalance_" + fault.split("_", 1)[1]
            elif token == "Misalign":
                detail = "misalign_" + fault.split("_", 1)[1]
            else:
                detail = fault
            return SourceMetadata(stem, label, detail)

    raise ValueError(f"Cannot infer label from source name: {stem}")


def slug_channel_name(channel_type: str, channel_name: str) -> str:
    mod = re.search(r"Mod(\d+)", channel_name)
    ai = re.search(r"(ai\d+)$", channel_name)
    if mod and ai:
        key = (channel_type, f"Mod{mod.group(1)}", ai.group(1))
        if key in TDMS_COLUMN_BY_CHANNEL:
            return TDMS_COLUMN_BY_CHANNEL[key]

    type_prefix = {
        "Temperature": "temperatura",
        "Current": "corrente",
    }[channel_type]
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", channel_name).strip("_").lower()
    return f"{type_prefix}_{suffix}"


def load_tdms_channels(path: Path) -> list[TdmsChannel]:
    channels: list[TdmsChannel] = []
    with TdmsFile.open(path) as tdms:
        for group in tdms.groups():
            for channel in group.channels():
                props = channel.properties
                channel_type = props.get("DAC~Channel~Type")
                if channel_type not in {"Temperature", "Current"}:
                    continue

                increment = props.get("wf_increment")
                if increment is None or len(channel) == 0:
                    continue

                start_offset = float(props.get("wf_start_offset", 0.0))
                unit = str(props.get("unit_string", ""))
                values = np.asarray(channel[:], dtype=np.float64)
                channels.append(
                    TdmsChannel(
                        column_name=slug_channel_name(channel_type, channel.name),
                        channel_type=channel_type,
                        unit=unit,
                        values=values,
                        start_offset=start_offset,
                        increment=float(increment),
                    )
                )

    if not channels:
        raise ValueError(f"{path.name}: no Temperature or Current channels found")

    return channels


def discover_tdms_columns(paths: Iterable[Path]) -> list[str]:
    columns: list[str] = []
    for path in paths:
        with TdmsFile.open(path) as tdms:
            for group in tdms.groups():
                for channel in group.channels():
                    channel_type = channel.properties.get("DAC~Channel~Type")
                    if channel_type in {"Temperature", "Current"}:
                        column_name = slug_channel_name(channel_type, channel.name)
                        if column_name not in columns:
                            columns.append(column_name)
    for column in TDMS_MEASUREMENT_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def pair_sources(input_dir: Path) -> list[tuple[str, Path, Path]]:
    mat_files = {path.stem: path for path in input_dir.glob("*.mat")}
    tdms_files = {path.stem: path for path in input_dir.glob("*.tdms")}

    missing_mat = sorted(set(tdms_files) - set(mat_files))
    missing_tdms = sorted(set(mat_files) - set(tdms_files))
    if missing_mat or missing_tdms:
        details = []
        if missing_mat:
            details.append(f"missing MAT for: {', '.join(missing_mat)}")
        if missing_tdms:
            details.append(f"missing TDMS for: {', '.join(missing_tdms)}")
        raise ValueError("; ".join(details))

    return [(stem, mat_files[stem], tdms_files[stem]) for stem in sorted(mat_files)]


def make_float_chunk_frame(
    metadata: SourceMetadata,
    mat_signal: MatSignal,
    tdms_channels: list[TdmsChannel],
    tdms_columns: list[str],
    start: int,
    stop: int,
) -> pd.DataFrame:
    sample_index = np.arange(start, stop, dtype=np.int64)
    time_s = mat_signal.start_value + sample_index * mat_signal.increment
    data: dict[str, Any] = {
        "sample_index": sample_index,
        "time_s": time_s,
        "label": metadata.label,
        "fault_detail": metadata.fault_detail,
    }

    vibration = mat_signal.values[start:stop]
    for idx in range(vibration.shape[1]):
        data[VIBRATION_COLUMNS[idx]] = vibration[:, idx]

    chunk_len = stop - start
    for column_name in tdms_columns:
        data[column_name] = np.full(chunk_len, np.nan, dtype=np.float64)
        data[f"{column_name}_valida"] = np.zeros(chunk_len, dtype=bool)

    for tdms_channel in tdms_channels:
        tdms_time = (
            tdms_channel.start_offset
            + np.arange(tdms_channel.sample_count, dtype=np.float64) * tdms_channel.increment
        )
        interpolated = np.interp(
            time_s,
            tdms_time,
            tdms_channel.values,
            left=np.nan,
            right=np.nan,
        )
        data[tdms_channel.column_name] = interpolated
        data[f"{tdms_channel.column_name}_valida"] = ~np.isnan(interpolated)

    return pd.DataFrame(data)


def update_max_abs(max_abs: dict[str, float], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    for column in MEASUREMENT_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        if np.isnan(values).all():
            continue
        current = float(np.nanmax(np.abs(values)))
        if current > max_abs[column]:
            max_abs[column] = current


def quantize_q15(frame: pd.DataFrame, max_abs: dict[str, float]) -> pd.DataFrame:
    q15_frame = frame[["label", "fault_detail"]].copy()
    for column in MEASUREMENT_COLUMNS:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        scale = max_abs[column] if max_abs[column] > 0.0 else 1.0
        normalized = values / scale
        quantized = np.rint(normalized * Q15_SCALE)
        quantized = np.nan_to_num(quantized, nan=0.0, posinf=32767.0, neginf=-32768.0)
        q15_frame[column] = np.clip(quantized, -32768, 32767).astype(np.int16)

    return q15_frame


def drop_missing_samples(frame: pd.DataFrame) -> pd.DataFrame:
    valid_rows = frame[TDMS_VALIDITY_COLUMNS].all(axis=1)
    return frame.loc[valid_rows].copy()


def report_row(
    metadata: SourceMetadata,
    mat_path: Path,
    tdms_path: Path,
    mat_signal: MatSignal,
    tdms_channels: list[TdmsChannel],
) -> dict[str, Any]:
    tdms_counts = {channel.column_name: channel.sample_count for channel in tdms_channels}
    tdms_increments = {channel.column_name: channel.increment for channel in tdms_channels}
    tdms_durations = {channel.column_name: channel.duration_s for channel in tdms_channels}
    max_tdms_duration = max(tdms_durations.values())

    return {
        "source_id": metadata.source_id,
        "label": metadata.label,
        "fault_detail": metadata.fault_detail,
        "source_mat": mat_path.name,
        "source_tdms": tdms_path.name,
        "mat_samples": mat_signal.sample_count,
        "mat_channels": mat_signal.values.shape[1],
        "mat_increment_s": mat_signal.increment,
        "mat_duration_s": mat_signal.duration_s,
        "tdms_channels": ";".join(channel.column_name for channel in tdms_channels),
        "tdms_channel_types": ";".join(
            f"{channel.column_name}:{channel.channel_type}" for channel in tdms_channels
        ),
        "tdms_units": ";".join(f"{channel.column_name}:{channel.unit}" for channel in tdms_channels),
        "tdms_samples_by_channel": ";".join(
            f"{name}:{count}" for name, count in sorted(tdms_counts.items())
        ),
        "tdms_increment_s_by_channel": ";".join(
            f"{name}:{value}" for name, value in sorted(tdms_increments.items())
        ),
        "tdms_duration_s_by_channel": ";".join(
            f"{name}:{value}" for name, value in sorted(tdms_durations.items())
        ),
        "duration_delta_s": mat_signal.duration_s - max_tdms_duration,
    }


def write_report(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No report rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_scale_report(path: Path, max_abs: dict[str, float]) -> None:
    rows = []
    for column in MEASUREMENT_COLUMNS:
        maximum = max_abs[column]
        rows.append(
            {
                "column": column,
                "max_abs_original": maximum,
                "normalization_factor": maximum if maximum > 0.0 else 1.0,
                "q_format": "int16_q1_15",
                "decode_normalized": f"{column}_q15 / 32768",
            }
        )
    write_report(path, rows)


def build_dataset(args: argparse.Namespace) -> None:
    pairs = pair_sources(args.input_dir)
    if args.limit_files is not None:
        pairs = pairs[: args.limit_files]
    if not pairs:
        raise ValueError(f"No paired MAT/TDMS sources found in {args.input_dir}")

    tdms_columns = discover_tdms_columns(tdms_path for _, _, tdms_path in pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    print("Pass 1/2: discovering global max_abs scales")
    max_abs = {column: 0.0 for column in MEASUREMENT_COLUMNS}
    report_rows: list[dict[str, Any]] = []
    for index, (stem, mat_path, tdms_path) in enumerate(pairs, start=1):
        metadata = parse_source_metadata(stem)
        print(f"[scale {index}/{len(pairs)}] {stem}")
        mat_signal = load_mat_signal(mat_path)
        if mat_signal.values.shape[1] != len(VIBRATION_COLUMNS):
            raise ValueError(
                f"{mat_path.name}: expected {len(VIBRATION_COLUMNS)} vibration channels, "
                f"got {mat_signal.values.shape[1]}"
            )
        tdms_channels = load_tdms_channels(tdms_path)
        report_rows.append(report_row(metadata, mat_path, tdms_path, mat_signal, tdms_channels))

        for start in range(0, mat_signal.sample_count, args.chunk_size):
            stop = min(start + args.chunk_size, mat_signal.sample_count)
            frame = make_float_chunk_frame(
                metadata,
                mat_signal,
                tdms_channels,
                tdms_columns,
                start,
                stop,
            )
            frame = drop_missing_samples(frame)
            update_max_abs(max_abs, frame)

    write_scale_report(args.scale_report, max_abs)

    print("Pass 2/2: writing Q1.15 Parquet")
    writer: pq.ParquetWriter | None = None
    preview_frames: list[pd.DataFrame] = []
    preview_remaining = max(0, args.preview_rows)
    total_rows = 0

    try:
        for index, (stem, mat_path, tdms_path) in enumerate(pairs, start=1):
            metadata = parse_source_metadata(stem)
            print(f"[write {index}/{len(pairs)}] {stem}")
            mat_signal = load_mat_signal(mat_path)
            tdms_channels = load_tdms_channels(tdms_path)

            for start in range(0, mat_signal.sample_count, args.chunk_size):
                stop = min(start + args.chunk_size, mat_signal.sample_count)
                frame = make_float_chunk_frame(
                    metadata,
                    mat_signal,
                    tdms_channels,
                    tdms_columns,
                    start,
                    stop,
                )
                frame = drop_missing_samples(frame)
                if frame.empty:
                    continue

                q15_frame = quantize_q15(frame, max_abs)
                table = pa.Table.from_pandas(q15_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, table.schema)
                writer.write_table(table)

                if preview_remaining > 0:
                    preview_frames.append(q15_frame.head(preview_remaining))
                    preview_remaining -= min(preview_remaining, len(q15_frame))
                total_rows += len(q15_frame)

    finally:
        if writer is not None:
            writer.close()

    write_report(args.report, report_rows)
    if args.preview_csv:
        args.preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()
        preview.to_csv(args.preview_csv, index=False)

    pq.ParquetFile(args.output)
    print(f"Processed {len(pairs)} source pairs")
    print(f"Wrote {total_rows} rows to {args.output}")
    print(f"Wrote report to {args.report}")
    print(f"Wrote scale report to {args.scale_report}")
    print(f"Wrote preview to {args.preview_csv}")


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
