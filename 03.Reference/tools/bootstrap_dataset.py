"""One-command setup: downloads the raw dataset from Mendeley, extracts
it, and runs the processing chain notebooks/analise_smma.ipynb needs
(train/val/test split, k-fold groups, MLP v2 features, and the CNN's
low-frequency spectrogram -- each with its dense-`operacao_normal`
variant). Every step is skipped if its output already exists, so
re-running after an interruption only redoes what's missing; pass
--force to rebuild everything from scratch.

The Mendeley download is a zip of zips: one long-named top-level folder
(itself close to Windows' 260-character path limit once combined with
an already-deep project path) containing acoustic.zip, current,temp.zip
and vibration.zip. Only the latter two are used here -- this pipeline
has no acoustic-sensor feature -- and both are flat internally (every
.mat/.tdms sits at their root, no further nesting). Extraction pulls
each of those two straight out of the outer zip, then flattens their
contents directly into RAW_DATASET_DIR by filename -- the flat layout
dataset/paths.py and every tools/build_*.py script expect -- so the
long outer folder name is never recreated on disk.

Other tools/build_*.py scripts (v1 features, native-resolution or
4-channel spectrograms, etc.) cover configurations that were explored
but not adopted -- see this package's README -- and are intentionally
not part of this default chain; run them directly if you need those
comparisons too.

Usage (from 03.Reference):
    python -m tools.bootstrap_dataset [--force]
"""

import argparse
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from dataset.paths import DATASET_DIR, EDA_OUTPUT_DIR, PROJECT_ROOT, RAW_DATASET_DIR

DATASET_URL = "https://data.mendeley.com/public-api/zip/ztmf3m7h5x/download/6"
EXPECTED_SOURCE_COUNT = 45  # see notebooks/analise_smma.ipynb's dataset table
REFERENCE_ROOT = PROJECT_ROOT / "03.Reference"
RAW_ARCHIVE_EXTENSIONS = {".mat", ".tdms"}
# Sub-archives inside the outer download; "acoustic.zip" also exists but
# is never extracted -- nothing in this pipeline reads acoustic data.
INNER_ARCHIVE_NAMES = ["vibration.zip", "current,temp.zip"]
DOWNLOAD_TIMEOUT_S = 60  # per socket read, not for the whole transfer


def raw_dataset_present():
    return len(list(RAW_DATASET_DIR.glob("*.mat"))) >= EXPECTED_SOURCE_COUNT


def _download(url, destination):
    print(f"Downloading dataset from {url}\n  -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    written = 0
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
        total = response.getheader("Content-Length")
        total = int(total) if total else None
        last_report = time.time()
        with destination.open("wb") as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                written += len(chunk)
                if time.time() - last_report > 2:
                    progress = f"{written / 1e6:,.0f} MB"
                    if total:
                        progress += f" / {total / 1e6:,.0f} MB ({100 * written / total:.0f}%)"
                    print(f"  {progress}")
                    last_report = time.time()
    print(f"Download complete: {written / 1e6:,.0f} MB")


def _extract_member(outer_zip_path, member_basename, destination_path):
    """Streams the single member of outer_zip_path named member_basename
    (matched by its own filename, ignoring the folder it's nested under)
    out to destination_path."""
    with zipfile.ZipFile(outer_zip_path) as archive:
        member = next((n for n in archive.namelist() if Path(n).name == member_basename), None)
        if member is None:
            raise RuntimeError(f"{member_basename} not found inside {outer_zip_path.name}")
        with archive.open(member) as source, destination_path.open("wb") as out_file:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)


def _extract_flat(zip_path, destination_dir):
    """Extracts every .mat/.tdms member directly into destination_dir
    under its own filename, discarding any folder it's nested under --
    see this module's docstring."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.suffix.lower() not in RAW_ARCHIVE_EXTENSIONS:
                continue
            target = destination_dir / name.name
            if target.exists():
                continue
            with archive.open(info) as source, target.open("wb") as out_file:
                out_file.write(source.read())
            extracted += 1
    print(f"Extracted {extracted} new .mat/.tdms files from {zip_path.name} to {destination_dir}")


def download_and_extract_raw_dataset(force=False):
    if raw_dataset_present() and not force:
        print(f"Raw dataset already present in {RAW_DATASET_DIR} (skipping download)")
        return

    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    outer_zip_path = RAW_DATASET_DIR / "_dataset_download.zip"
    inner_zip_paths = [RAW_DATASET_DIR / f"_inner_{name}" for name in INNER_ARCHIVE_NAMES]
    try:
        _download(DATASET_URL, outer_zip_path)
        if not zipfile.is_zipfile(outer_zip_path):
            raise RuntimeError(f"{outer_zip_path} is not a valid zip file (incomplete download?)")

        for name, inner_path in zip(INNER_ARCHIVE_NAMES, inner_zip_paths):
            print(f"Extracting {name} from the downloaded archive...")
            _extract_member(outer_zip_path, name, inner_path)
    finally:
        if outer_zip_path.exists():
            outer_zip_path.unlink()

    try:
        for inner_path in inner_zip_paths:
            _extract_flat(inner_path, RAW_DATASET_DIR)
    finally:
        for inner_path in inner_zip_paths:
            if inner_path.exists():
                inner_path.unlink()

    found = len(list(RAW_DATASET_DIR.glob("*.mat")))
    if found < EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"Only {found} .mat files found after extraction, expected "
            f"{EXPECTED_SOURCE_COUNT} -- the archive layout may have changed."
        )


def _run_step(description, output_path, args, force):
    if output_path.exists() and not force:
        print(f"[skip] {description} (already exists: {output_path.name})")
        return
    print(f"[run]  {description}")
    t0 = time.time()
    subprocess.run([sys.executable, *args], cwd=REFERENCE_ROOT, check=True)
    print(f"       done in {time.time() - t0:.0f}s")


def build_processed_dataset(force=False):
    _run_step(
        "consolidated Q1.15 parquet", DATASET_DIR / "motor_measurements_q15.parquet",
        ["build_dataset.py"], force,
    )
    _run_step(
        "train/val/test split", EDA_OUTPUT_DIR / "train_val_test_split.csv",
        ["-m", "tools.build_train_val_test_split"], force,
    )
    _run_step(
        "k-fold groups", EDA_OUTPUT_DIR / "folds.csv",
        ["-m", "tools.build_folds"], force,
    )
    _run_step(
        "MLP v2 features (base)", DATASET_DIR / "features_dataset_v2.parquet",
        ["-m", "tools.build_features_dataset_v2"], force,
    )
    _run_step(
        "MLP v2 features (dense operacao_normal)", DATASET_DIR / "features_dataset_v2_dense_normal.parquet",
        ["-m", "tools.build_features_dataset_dense_normal", "--pipeline", "v2"], force,
    )
    _run_step(
        "CNN low-freq spectrogram, mancal A (base)", DATASET_DIR / "spectrogram_dataset_lowfreq_ch0-1.npz",
        ["-m", "tools.build_spectrogram_dataset_lowfreq"], force,
    )
    _run_step(
        "CNN low-freq spectrogram, mancal A (dense operacao_normal)",
        DATASET_DIR / "spectrogram_dataset_lowfreq_ch0-1_dense_normal.npz",
        ["-m", "tools.build_spectrogram_dataset_lowfreq_dense_normal"], force,
    )


def ensure_dataset_ready(force=False):
    """Single entry point for notebooks/scripts: guarantees every file
    notebooks/analise_smma.ipynb reads exists, downloading and building
    whatever is missing. Safe to call unconditionally on every run --
    each step no-ops once its output is already on disk.
    """
    download_and_extract_raw_dataset(force=force)
    build_processed_dataset(force=force)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="re-download and rebuild every step even if its output already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dataset_ready(force=args.force)


if __name__ == "__main__":
    main()
