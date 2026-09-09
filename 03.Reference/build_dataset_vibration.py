#!/usr/bin/env python3
"""Build a fixed-point vibration-only motor measurements dataset from MAT files."""

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
from scipy.io import loadmat


DEFAULT_INPUT_DIR = Path("07.Datasets")
DEFAULT_OUTPUT_DIR = Path("07.Datasets/processed")
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "dataset_report.csv"
DEFAULT_SCALE_REPORT = DEFAULT_OUTPUT_DIR / "dataset_scale_report.csv"
DEFAULT_PREVIEW = DEFAULT_OUTPUT_DIR / "dataset_preview.csv"

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

Q_FORMATS = {
    "q1.15": {
        "scale": 32768.0,
        "clip_min": -32768,
        "clip_max": 32767,
        "dtype": np.int16,
        "report_name": "int16_q1_15",
        "output_suffix": "q115",
    },
    "q1.7": {
        "scale": 128.0,
        "clip_min": -128,
        "clip_max": 127,
        "dtype": np.int8,
        "report_name": "int8_q1_7",
        "output_suffix": "q17",
    },
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a vibration-only Parquet dataset from MAT files, subtracting "
            "global per-channel means before fixed-point quantization."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output Parquet path. Defaults to "
            "07.Datasets/processed/motor_vibration_q15.parquet or "
            "motor_vibration_q17.parquet according to --q-format."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scale-report", type=Path, default=DEFAULT_SCALE_REPORT)
    parser.add_argument("--preview-csv", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=1000,
        help="Number of first quantized rows to save in the preview CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows per processing/write chunk.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Process only the first N MAT sources; useful for smoke tests.",
    )
    parser.add_argument(
        "--q-format",
        choices=sorted(Q_FORMATS),
        default="q1.15",
        help="Fixed-point quantization format to use. Defaults to q1.15.",
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


def discover_mat_sources(input_dir: Path) -> list[tuple[str, Path]]:
    return [(path.stem, path) for path in sorted(input_dir.glob("*.mat"))]


def validate_vibration_shape(path: Path, mat_signal: MatSignal) -> None:
    if mat_signal.values.shape[1] != len(VIBRATION_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(VIBRATION_COLUMNS)} vibration channels, "
            f"got {mat_signal.values.shape[1]}"
        )


def iter_chunk_bounds(sample_count: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    for start in range(0, sample_count, chunk_size):
        yield start, min(start + chunk_size, sample_count)


def update_sums(
    sums: dict[str, float],
    counts: dict[str, int],
    mat_signal: MatSignal,
    start: int,
    stop: int,
) -> None:
    vibration = mat_signal.values[start:stop]
    for idx, column in enumerate(VIBRATION_COLUMNS):
        values = vibration[:, idx]
        valid = values[~np.isnan(values)]
        sums[column] += float(valid.sum())
        counts[column] += int(valid.shape[0])


def compute_means(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    means = {}
    for column in VIBRATION_COLUMNS:
        if counts[column] == 0:
            raise ValueError(f"No valid samples found for {column}")
        means[column] = sums[column] / counts[column]
    return means


def update_centered_max_abs(
    max_abs: dict[str, float],
    means: dict[str, float],
    mat_signal: MatSignal,
    start: int,
    stop: int,
) -> None:
    vibration = mat_signal.values[start:stop]
    for idx, column in enumerate(VIBRATION_COLUMNS):
        centered = vibration[:, idx] - means[column]
        if np.isnan(centered).all():
            continue
        current = float(np.nanmax(np.abs(centered)))
        if current > max_abs[column]:
            max_abs[column] = current


def quantize_chunk(
    metadata: SourceMetadata,
    mat_signal: MatSignal,
    means: dict[str, float],
    max_abs: dict[str, float],
    q_format: str,
    start: int,
    stop: int,
) -> pd.DataFrame:
    q_config = Q_FORMATS[q_format]
    q_frame = pd.DataFrame(
        {
            "label": np.full(stop - start, metadata.label, dtype=object),
            "fault_detail": np.full(stop - start, metadata.fault_detail, dtype=object),
        }
    )

    vibration = mat_signal.values[start:stop]
    for idx, column in enumerate(VIBRATION_COLUMNS):
        scale = max_abs[column] if max_abs[column] > 0.0 else 1.0
        centered = vibration[:, idx] - means[column]
        normalized = centered / scale
        quantized = np.rint(normalized * q_config["scale"])
        quantized = np.nan_to_num(
            quantized,
            nan=0.0,
            posinf=float(q_config["clip_max"]),
            neginf=float(q_config["clip_min"]),
        )
        q_frame[column] = np.clip(
            quantized,
            q_config["clip_min"],
            q_config["clip_max"],
        ).astype(q_config["dtype"])

    return q_frame


def report_row(metadata: SourceMetadata, mat_path: Path, mat_signal: MatSignal) -> dict[str, Any]:
    return {
        "source_id": metadata.source_id,
        "label": metadata.label,
        "fault_detail": metadata.fault_detail,
        "source_mat": mat_path.name,
        "mat_samples": mat_signal.sample_count,
        "mat_channels": mat_signal.values.shape[1],
        "mat_increment_s": mat_signal.increment,
        "mat_duration_s": mat_signal.duration_s,
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


def write_scale_report(
    path: Path,
    means: dict[str, float],
    max_abs: dict[str, float],
    q_format: str,
) -> None:
    q_config = Q_FORMATS[q_format]
    rows = []
    for column in VIBRATION_COLUMNS:
        maximum = max_abs[column]
        rows.append(
            {
                "column": column,
                "mean_subtracted": means[column],
                "max_abs_centered": maximum,
                "normalization_factor": maximum if maximum > 0.0 else 1.0,
                "q_format": q_config["report_name"],
                "decode_normalized": f"{column}_{q_config['output_suffix']} / {int(q_config['scale'])}",
            }
        )
    write_report(path, rows)


def default_output_path(q_format: str) -> Path:
    suffix = Q_FORMATS[q_format]["output_suffix"]
    return DEFAULT_OUTPUT_DIR / f"motor_vibration_{suffix}.parquet"


def build_dataset(args: argparse.Namespace) -> None:
    sources = discover_mat_sources(args.input_dir)
    if args.limit_files is not None:
        sources = sources[: args.limit_files]
    if not sources:
        raise ValueError(f"No MAT sources found in {args.input_dir}")

    output = args.output or default_output_path(args.q_format)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    print("Pass 1/3: computing global vibration means")
    sums = {column: 0.0 for column in VIBRATION_COLUMNS}
    counts = {column: 0 for column in VIBRATION_COLUMNS}
    report_rows: list[dict[str, Any]] = []
    for index, (stem, mat_path) in enumerate(sources, start=1):
        metadata = parse_source_metadata(stem)
        print(f"[mean {index}/{len(sources)}] {stem}")
        mat_signal = load_mat_signal(mat_path)
        validate_vibration_shape(mat_path, mat_signal)
        report_rows.append(report_row(metadata, mat_path, mat_signal))

        for start, stop in iter_chunk_bounds(mat_signal.sample_count, args.chunk_size):
            update_sums(sums, counts, mat_signal, start, stop)

    means = compute_means(sums, counts)

    print("Pass 2/3: discovering centered max_abs scales")
    max_abs = {column: 0.0 for column in VIBRATION_COLUMNS}
    for index, (stem, mat_path) in enumerate(sources, start=1):
        print(f"[scale {index}/{len(sources)}] {stem}")
        mat_signal = load_mat_signal(mat_path)
        validate_vibration_shape(mat_path, mat_signal)

        for start, stop in iter_chunk_bounds(mat_signal.sample_count, args.chunk_size):
            update_centered_max_abs(max_abs, means, mat_signal, start, stop)

    write_scale_report(args.scale_report, means, max_abs, args.q_format)

    print(f"Pass 3/3: writing {args.q_format.upper()} Parquet")
    writer: pq.ParquetWriter | None = None
    preview_frames: list[pd.DataFrame] = []
    preview_remaining = max(0, args.preview_rows)
    total_rows = 0

    try:
        for index, (stem, mat_path) in enumerate(sources, start=1):
            metadata = parse_source_metadata(stem)
            print(f"[write {index}/{len(sources)}] {stem}")
            mat_signal = load_mat_signal(mat_path)
            validate_vibration_shape(mat_path, mat_signal)

            for start, stop in iter_chunk_bounds(mat_signal.sample_count, args.chunk_size):
                q_frame = quantize_chunk(
                    metadata,
                    mat_signal,
                    means,
                    max_abs,
                    args.q_format,
                    start,
                    stop,
                )
                if q_frame.empty:
                    continue

                table = pa.Table.from_pandas(q_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema)
                writer.write_table(table)

                if preview_remaining > 0:
                    preview_frames.append(q_frame.head(preview_remaining))
                    preview_remaining -= min(preview_remaining, len(q_frame))
                total_rows += len(q_frame)

    finally:
        if writer is not None:
            writer.close()

    write_report(args.report, report_rows)
    if args.preview_csv:
        args.preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview = pd.concat(preview_frames, ignore_index=True) if preview_frames else pd.DataFrame()
        preview.to_csv(args.preview_csv, index=False)

    pq.ParquetFile(output)
    print(f"Processed {len(sources)} MAT sources")
    print(f"Wrote {total_rows} rows to {output}")
    print(f"Wrote report to {args.report}")
    print(f"Wrote scale report to {args.scale_report}")
    print(f"Wrote preview to {args.preview_csv}")


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
