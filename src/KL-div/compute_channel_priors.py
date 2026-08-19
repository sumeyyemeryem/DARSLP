"""
Compute Per-Channel Priors — DARSLP Phase 2
============================================
Analyses the per-channel distribution of DisentangledAE latent encodings
extracted from the training set. For each latent dimension the script computes:
  mean, std, min, max, Shannon entropy (fixed-width bins), normalised entropy, IQR

The resulting CSV is then passed to save_channel_priors.py to produce the
.npy/.npz prior file consumed by ChannelKLLoss during Phase 2 training.

This script answers the question "what does the ground-truth latent space look
like per channel?" and is the first step in the channel-aware KL regularization
pipeline described in the DARSLP paper.

Typical workflow:
  1. python precompute_encodings.py  → generates per-sample .npy encoding files
  2. python compute_channel_priors.py  (edit csv_path at the bottom)
     → produces stats-all-final-disentangled-wface-3d.csv
  3. python save_channel_priors.py
     → produces data/channel_priors/channel_priors_<dataset>_<dim>dim.npz

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def compute_entropy_and_iqr(data, bins):
    hist, _ = np.histogram(data, bins=bins, density=True)
    p = hist / hist.sum() if hist.sum() > 0 else np.ones_like(hist) / len(hist)
    H   = entropy(p, base=2)
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    return H, IQR


def compute_channel_stats(file_path, bin_width=0.1):
    """
    Load a CSV of per-frame latent encodings and compute per-channel statistics.
    Saves results to stats-all-final-disentangled-wface-3d.csv.

    Args:
        file_path: CSV where each row is one frame; first column is filename,
                   remaining columns are latent dimensions.
        bin_width: histogram bin width for entropy computation.
    """
    df         = pd.read_csv(file_path)
    df_numeric = df.iloc[:, 1:]
    num_dims   = df_numeric.shape[1]

    stats_list = []
    for col_idx in range(num_dims):
        data = df_numeric.iloc[:, col_idx].dropna()
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        H, IQR = compute_entropy_and_iqr(data, bins)
        max_entropy       = np.log2(len(bins)) if len(bins) > 1 else 1
        normalised_entropy = H / max_entropy
        stats_list.append([
            col_idx, H, normalised_entropy, IQR,
            data.min(), data.max(), data.mean(), data.std()
        ])

    stats_df = pd.DataFrame(stats_list, columns=[
        "Dimension", "Entropy", "Normalized Entropy", "IQR",
        "Min", "Max", "Mean", "StdDev"
    ])
    out_path = "stats-all-final-disentangled-wface-3d.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Saved per-channel stats -> {out_path}")
    return stats_df


if __name__ == "__main__":
    # Edit this path to point to your training-set encoding CSV
    # (CSV with one row per frame, first column = filename, rest = latent dims)
    csv_path = "path/to/train_pose_encodings.csv"
    compute_channel_stats(csv_path, bin_width=0.1)
