#!/usr/bin/env python3
"""Preview paired MAT and TDMS source files without consolidating them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import loadmat

from build_dataset import (
    DEFAULT_INPUT_DIR,
    VIBRATION_COLUMNS,
    get_field,
    load_mat_signal,
    pair_sources,
    parse_source_metadata,
    slug_channel_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview MAT vibration and TDMS current/temperature source files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--source-id",
        default=None,
        help="Source stem to inspect, for example 0Nm_Normal. Defaults to the first pair.",
    )
    parser.add_argument("--rows", type=int, default=8, help="Number of sample rows to print.")
    return parser.parse_args()


def summarize_mat(path: Path, rows: int) -> None:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    signal = raw["Signal"]
    x_values = get_field(signal, "x_values")
    y_values = get_field(signal, "y_values")
    mat_signal = load_mat_signal(path)

    print("\nMAT")
    print(f"file: {path}")
    print(f"top-level keys: {sorted(k for k in raw if not k.startswith('__'))}")
    print(f"signal fields: {sorted(name for name in dir(signal) if not name.startswith('_'))}")
    print(f"x start_value: {mat_signal.start_value}")
    print(f"x increment_s: {mat_signal.increment}")
    print(f"x sample_rate_hz: {1.0 / mat_signal.increment}")
    print(f"x number_of_values: {mat_signal.sample_count}")
    print(f"y values shape: {mat_signal.values.shape}")
    print(f"y dtype: {mat_signal.values.dtype}")

    for field_name in ("quantity", "function_record"):
        try:
            field: Any = get_field(y_values if field_name == "quantity" else signal, field_name)
            print(f"{field_name} type: {type(field).__name__}")
        except Exception:
            pass

    preview = pd.DataFrame(
        {
            "sample_index": np.arange(min(rows, mat_signal.sample_count)),
            "time_s": mat_signal.start_value
            + np.arange(min(rows, mat_signal.sample_count)) * mat_signal.increment,
            **{
                VIBRATION_COLUMNS[index]: mat_signal.values[:rows, index]
                for index in range(mat_signal.values.shape[1])
            },
        }
    )
    print("\nMAT sample values")
    print(preview.to_string(index=False))


def summarize_tdms(path: Path, rows: int) -> None:
    print("\nTDMS")
    print(f"file: {path}")
    with TdmsFile.open(path) as tdms:
        for group in tdms.groups():
            print(f"\ngroup: {group.name}")
            for channel in group.channels():
                props = channel.properties
                channel_type = props.get("DAC~Channel~Type", "")
                unit = props.get("unit_string", "")
                column = (
                    slug_channel_name(channel_type, channel.name)
                    if channel_type in {"Temperature", "Current"}
                    else ""
                )
                increment = props.get("wf_increment")
                sample_rate = None if increment in (None, 0) else 1.0 / float(increment)
                print(
                    "  "
                    f"channel={channel.name} "
                    f"type={channel_type or '-'} "
                    f"unit={unit or '-'} "
                    f"column={column or '-'} "
                    f"samples={len(channel)} "
                    f"wf_increment={increment} "
                    f"sample_rate_hz={sample_rate}"
                )
                interesting_props = {
                    key: props[key]
                    for key in (
                        "wf_xname",
                        "wf_xunit_string",
                        "wf_samples",
                        "wf_start_offset",
                        "wf_start_time",
                    )
                    if key in props
                }
                print(f"    props: {interesting_props}")
                if len(channel):
                    values = np.asarray(channel[: min(rows, len(channel))], dtype=np.float64)
                    print(f"    first_values: {values.tolist()}")


def main() -> None:
    args = parse_args()
    pairs = pair_sources(args.input_dir)
    by_source = {stem: (mat_path, tdms_path) for stem, mat_path, tdms_path in pairs}
    source_id = args.source_id or pairs[0][0]
    if source_id not in by_source:
        available = ", ".join(sorted(by_source))
        raise SystemExit(f"Unknown source-id {source_id!r}. Available: {available}")

    metadata = parse_source_metadata(source_id)
    mat_path, tdms_path = by_source[source_id]
    print(f"source_id: {metadata.source_id}")
    print(f"label: {metadata.label}")
    print(f"fault_detail: {metadata.fault_detail}")
    print(f"load_nm: {metadata.load_nm}")
    summarize_mat(mat_path, args.rows)
    summarize_tdms(tdms_path, args.rows)


if __name__ == "__main__":
    main()
