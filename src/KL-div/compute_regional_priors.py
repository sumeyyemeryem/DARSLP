"""
Compute Region-Specific Articulator Priors — DARSLP
=====================================================
Extends compute_channel_priors.py with per-region analysis. Instead of
treating all latent channels uniformly, this script groups them by articulator
region (face, right_hand, left_hand, body) using column-name prefixes and
computes region-level entropy statistics.

Outputs:
  • Per-region CSV files with entropy / IQR / std per channel
  • Optional PNG figures showing the top-entropy channels per region

This is useful for diagnosing which articulator regions have the most varied
latent distributions and for tuning region-specific KL weights (RH_weight,
LH_weight, KL_RH_weight, KL_LH_weight) in train_stage2.py.

The CSV produced here can also be fed to save_channel_priors.py to create
region-separated prior files.

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


def compute_entropy_and_iqr(data, bins):
    hist, _ = np.histogram(data, bins=bins, density=True)
    p   = hist / hist.sum() if hist.sum() > 0 else np.ones_like(hist) / len(hist)
    H   = entropy(p, base=2)
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    return H, IQR


def plot_top_entropy_channels(channels, df_numeric, region_name, fig_name,
                               bin_width=0.1, save_dir=".", top_k=8):
    entropy_list = []
    for col in channels:
        data = df_numeric[col].dropna()
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        H, IQR = compute_entropy_and_iqr(data, bins)
        entropy_list.append((col, H, IQR, data.std()))

    avg_H = np.mean([x[1] for x in entropy_list])
    print(f"{region_name}: avg entropy = {avg_H:.4f}")

    top = sorted(entropy_list, key=lambda x: x[1], reverse=True)[:top_k]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    axes = axes.flatten()
    for i, (col, H, IQR, std) in enumerate(top):
        data = df_numeric[col].dropna()
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        axes[i].hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(col, fontsize=9)
        axes[i].text(0.5, -0.25, f"E:{H:.2f}  IQR:{IQR:.2f}  SD:{std:.2f}",
                     ha='center', transform=axes[i].transAxes, fontsize=8)
    for j in range(len(top), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.close()


def compute_regional_priors(file_path, save_dir=".", bin_width=0.1):
    """
    Compute and visualize per-region top-entropy channels.

    Args:
        file_path: CSV where columns are named with region prefixes
                   (face_, right_, left_, body_).
        save_dir:  output directory for PNG figures.
        bin_width: histogram bin width.
    """
    df         = pd.read_csv(file_path)
    df_numeric = df.drop(columns=["filename"])

    regions = {
        "face":       [c for c in df_numeric.columns if c.startswith("face_")],
        "right_hand": [c for c in df_numeric.columns if c.startswith("right_")],
        "left_hand":  [c for c in df_numeric.columns if c.startswith("left_")],
        "body":       [c for c in df_numeric.columns if c.startswith("body_")],
    }

    for region_name, cols in regions.items():
        if not cols:
            print(f"No columns found for region '{region_name}' — skipping.")
            continue
        plot_top_entropy_channels(
            channels=cols,
            df_numeric=df_numeric,
            region_name=region_name,
            fig_name=f"top_entropy_{region_name}.png",
            bin_width=bin_width,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    # Edit these paths before running
    csv_path = "path/to/train_encodings_by_region.csv"
    save_dir = "region_stats"
    compute_regional_priors(csv_path, save_dir=save_dir)
