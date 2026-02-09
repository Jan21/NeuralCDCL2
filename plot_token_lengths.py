"""Plot histograms of token lengths per example for each data file."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).parent / "output"
OUTPUT_DIR = Path(__file__).parent / "token_length_plots"

# Files to process (skip nested traces - different format, very large)
DATA_FILES = [
    "train.json",
    "iid_test.json",
    "iid_test_solve.json",
    "iid_test_ac.json",
    "iid_test_up.json",
    "ood_test.json",
    "ood_test_solve.json",
    "ood_test_ac.json",
    "ood_test_up.json",
]


def stream_token_lengths(filepath: Path) -> list[int]:
    """Stream a JSON file and return token lengths per example.

    Handles both small files (load entire JSON array) and large files
    (line-by-line streaming for train.json).
    """
    lengths = []
    file_size = filepath.stat().st_size

    if file_size > 500_000_000:  # > 500MB, stream line-by-line
        print(f"  Streaming (large file: {file_size / 1e9:.1f} GB)...")
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ("[\n", "]\n", "[", "]"):
                    continue
                try:
                    obj = json.loads(line)
                    lengths.append(len(obj["text"].split()))
                except (json.JSONDecodeError, KeyError):
                    continue
    else:
        print(f"  Loading ({file_size / 1e6:.1f} MB)...")
        with open(filepath, "r") as f:
            data = json.load(f)
        for obj in data:
            lengths.append(len(obj["text"].split()))

    return lengths


def plot_histogram(lengths: list[int], filename: str, output_path: Path):
    """Create and save a histogram of token lengths."""
    lengths_arr = np.array(lengths)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(lengths_arr, bins=80, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Token Length (per example)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Token Length Distribution — {filename}", fontsize=14)

    # Add stats text box
    stats_text = (
        f"N = {len(lengths_arr):,}\n"
        f"Mean = {lengths_arr.mean():.0f}\n"
        f"Median = {np.median(lengths_arr):.0f}\n"
        f"Min = {lengths_arr.min():,}\n"
        f"Max = {lengths_arr.max():,}\n"
        f"Std = {lengths_arr.std():.0f}"
    )
    ax.text(
        0.97, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for filename in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        print(f"Processing {filename}...")
        lengths = stream_token_lengths(filepath)
        if not lengths:
            print(f"  No examples found, skipping.")
            continue

        out_path = OUTPUT_DIR / f"{filepath.stem}_token_lengths.png"
        plot_histogram(lengths, filename, out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
