"""Filesystem locations for the consolidated motor-measurements dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_DIR = PROJECT_ROOT / "07.Datasets"
DATASET_DIR = RAW_DATASET_DIR / "processed"

DATASET_PARQUET = DATASET_DIR / "motor_measurements_q15.parquet"
SOURCE_REPORT_CSV = DATASET_DIR / "dataset_report.csv"
SCALE_REPORT_CSV = DATASET_DIR / "dataset_scale_report.csv"

EDA_OUTPUT_DIR = DATASET_DIR / "eda"
