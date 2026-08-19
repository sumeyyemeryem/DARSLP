"""
Save Channel Priors to File — DARSLP Phase 2
=============================================
Reads the per-channel statistics CSV produced by compute_channel_priors.py
and saves the 'Mean' and 'StdDev' columns as a portable .npz file.

The resulting .npz file is the prior file consumed by ChannelKLLoss during
DARSLP Phase 2 training. Pass its path via the --prior_file argument of
train_stage2.py.

File format (.npz):
    Mean   — float32 array of shape (latent_dim,)
    StdDev — float32 array of shape (latent_dim,)

Usage:
    python save_channel_priors.py
    (edit the csv_file and npz_file paths at the bottom)

Reference: Taşyürek et al., "Disentangle and Regularize: Sign Language Production
with Articulator-Based Disentanglement and Channel-Aware Regularization," WACV 2026.
"""

import numpy as np
import pandas as pd
import torch


def save_priors_npz(csv_file, npz_file, delimiter=","):
    """
    Read Mean and StdDev from a stats CSV and save as a .npz prior file.

    Args:
        csv_file:  path to the CSV produced by compute_channel_priors.py
        npz_file:  output .npz path
        delimiter: CSV delimiter (default: comma)
    """
    df = pd.read_csv(csv_file, delimiter=delimiter)
    df.columns = df.columns.str.strip()

    mean_arr = torch.tensor(df['Mean'].values,   dtype=torch.float32).cpu().numpy()
    std_arr  = torch.tensor(df['StdDev'].values, dtype=torch.float32).cpu().numpy()

    np.savez(npz_file, Mean=mean_arr, StdDev=std_arr)
    print(f"Saved channel priors -> {npz_file}  (keys: ['Mean', 'StdDev'])")


def load_and_verify(path):
    """Quick sanity check: load the saved prior file and print shapes."""
    if path.endswith(".npz"):
        f = np.load(path)
        print(f"Mean:   {f['Mean'].shape}  {f['Mean'][:4]}")
        print(f"StdDev: {f['StdDev'].shape}  {f['StdDev'][:4]}")
    else:
        d = np.load(path, allow_pickle=True).item()
        print(f"Mean:   {d['Mean'].shape}")
        print(f"StdDev: {d['StdDev'].shape}")


if __name__ == "__main__":
    # Edit these paths to match your setup
    csv_file = "stats-all-final-disentangled-wface-3d.csv"
    npz_file = "data/channel_priors/channel_priors_phoenix_80dim.npz"

    save_priors_npz(csv_file, npz_file)
    load_and_verify(npz_file)
