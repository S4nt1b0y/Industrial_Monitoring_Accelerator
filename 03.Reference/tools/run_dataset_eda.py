"""Dataset EDA: class distribution, channel validity, and physical-unit stats.

Usage (from 03.Reference):
    python -m tools.run_dataset_eda
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import eda
from dataset.loader import MEASUREMENT_COLUMNS, load_scale_factors, open_connection
from dataset.paths import EDA_OUTPUT_DIR
from dataset.signal_params import FREQ_RESOLUTION_HZ, FS_HZ, N_FFT, WINDOW_DURATION_S


def print_signal_params():
    print("=== Signal parameters ===")
    print(f"fs (real dataset)       : {FS_HZ:,.0f} Hz")
    print(f"N (FFT, fixed by spec)  : {N_FFT}")
    print(f"window duration @ fs    : {WINDOW_DURATION_S * 1000:.3f} ms")
    print(f"freq. resolution (fs/N) : {FREQ_RESOLUTION_HZ:.1f} Hz/bin")
    print()


def plot_label_distribution(label_df, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(label_df["label"], label_df["rows"])
    ax.set_ylabel("rows")
    ax.set_title("Row count per class")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_validity_heatmap(validity_df, out_path):
    validity_columns = [c for c in validity_df.columns if c.endswith("_valida")]
    fig, ax = plt.subplots(figsize=(8, 6))
    data = validity_df[validity_columns].to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(validity_df)))
    ax.set_yticklabels(validity_df["fault_detail"])
    ax.set_xticks(range(len(validity_columns)))
    ax.set_xticklabels(validity_columns, rotation=45, ha="right")
    ax.set_title("Channel validity fraction by fault_detail (1.0 = fully recorded)")
    fig.colorbar(im, ax=ax, label="fraction valid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_channel_rms_by_label(stats_df, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, column in zip(axes.flat, MEASUREMENT_COLUMNS):
        ax.bar(stats_df["label"], stats_df[f"{column}_rms"])
        ax.set_title(column, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("RMS (physical units) per channel, by class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_signal_params()

    con = open_connection()
    scale_factors = load_scale_factors()

    print("Computing class distribution...")
    class_df = eda.class_distribution(con)
    class_df.to_csv(EDA_OUTPUT_DIR / "class_distribution.csv", index=False)

    label_df = eda.label_distribution(con)
    label_df.to_csv(EDA_OUTPUT_DIR / "label_distribution.csv", index=False)
    print(label_df.to_string(index=False))
    print()

    print("Computing channel validity by fault_detail...")
    validity_df = eda.validity_by_fault(con)
    validity_df.to_csv(EDA_OUTPUT_DIR / "validity_by_fault.csv", index=False)
    print(validity_df.to_string(index=False))
    print()

    print("Computing physical-unit channel stats by label (this scans the full dataset)...")
    stats_df = eda.channel_stats_by_label(con, scale_factors)
    stats_df.to_csv(EDA_OUTPUT_DIR / "channel_stats_by_label.csv", index=False)
    print(stats_df.to_string(index=False))
    print()

    plot_label_distribution(label_df, EDA_OUTPUT_DIR / "label_distribution.png")
    plot_validity_heatmap(validity_df, EDA_OUTPUT_DIR / "validity_by_fault.png")
    plot_channel_rms_by_label(stats_df, EDA_OUTPUT_DIR / "channel_rms_by_label.png")

    print(f"Wrote CSVs and plots to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
