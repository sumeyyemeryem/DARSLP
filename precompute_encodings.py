"""
Precompute Pose Encodings — DARSLP Stage 2 Preparation
=======================================================
Runs the trained Stage 1 autoencoder over a dataset split and saves the
resulting latent encodings as per-sample .npy files.

These files are the regression targets for DARSLPGenerator (Stage 2) training.
Precomputing them once avoids re-encoding on every training epoch.

Output per sample:
    <output_dir>/<sample_key>.npy   shape: (F, latent_dim)

    For DisentangledAE: concatenated (z_upper ‖ z_right ‖ z_left ‖ z_face).
    For StandardAE:     single unified latent vector.

Suggested output directory convention (used as default in train_stage2.py):
    data/pose_encodings/<dataset>_<split>_disentangled_<dim>dim/
    data/pose_encodings/<dataset>_<split>_standard_<dim>dim/

Example — PHOENIX-2014T train split (DisentangledAE, 80-dim):
    python precompute_encodings.py \\
        --model       disentangled \\
        --ae_ckpt     models/ae_phoenix_disentangled.pth \\
        --poses       /data/phoenix/train.pt \\
        --output_dir  data/pose_encodings/phoenix_train_disentangled_80dim \\
        --max_frame_len 300

Example — CSL-Daily dev split (DisentangledAE, 80-dim):
    python precompute_encodings.py \\
        --model       disentangled \\
        --ae_ckpt     models/ae_csl_disentangled.pth \\
        --poses       /data/csl/dev.pt \\
        --output_dir  data/pose_encodings/csl_dev_disentangled_80dim \\
        --max_frame_len 350

Example — PHOENIX-2014T train split (StandardAE, 80-dim, ablation):
    python precompute_encodings.py \\
        --model       standard \\
        --ae_ckpt     models/ae_phoenix_standard.pth \\
        --poses       /data/phoenix/train.pt \\
        --output_dir  data/pose_encodings/phoenix_train_standard_80dim \\
        --max_frame_len 300
"""

import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.data.SLPDataset import SLPDataset


def encode_disentangled(ae_model, poses, device):
    """Encode a batch with DisentangledAE → concatenated latent [B, F, latent_dim]."""
    poses = poses.to(device)
    with torch.no_grad():
        z_upper, z_right, z_left, z_face = ae_model.encode(poses)
    return torch.cat([z_upper, z_right, z_left, z_face], dim=-1).cpu()


def encode_standard(ae_model, poses, device):
    """Encode a batch with StandardAE → latent [B, F, latent_dim]."""
    poses = poses.to(device)
    with torch.no_grad():
        z = ae_model.encode(poses)
    return z.cpu()


def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    ae_model.to(device).eval()

    encode_fn = encode_disentangled if args.model == "disentangled" else encode_standard

    # Minimal dataset — no text embeddings needed, use a dummy path
    # SLPDataset only raises if the .npy is missing when __getitem__ is called.
    # We pass a dummy path and skip text loading by wrapping the collation.
    data = torch.load(args.poses, weights_only=False)
    keys = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    existing = set(os.listdir(args.output_dir))
    skipped  = 0
    processed = 0

    for i, key in enumerate(keys):
        out_file = os.path.join(args.output_dir, f"{key}.npy")

        if not args.overwrite and f"{key}.npy" in existing:
            skipped += 1
            continue

        poses_3d = data[key]["poses_3d"].numpy()   # (F, 178, 3)
        F = poses_3d.shape[0]
        # Truncate to max_frame_len if needed; no random drop for encodings
        if F > args.max_frame_len:
            poses_3d = poses_3d[:args.max_frame_len]

        poses_tensor = torch.tensor(poses_3d, dtype=torch.float32).unsqueeze(0)  # (1, F, 178, 3)
        encoding     = encode_fn(ae_model, poses_tensor, device)                 # (1, F, latent_dim)
        encoding_np  = encoding.squeeze(0).numpy()                               # (F, latent_dim)

        np.save(out_file, encoding_np)
        processed += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(keys):
            print(f"  [{i+1}/{len(keys)}]  saved: {processed}  skipped: {skipped}")

    print(f"\nDone. {processed} encodings saved to {args.output_dir}")
    if skipped:
        print(f"  ({skipped} already existed - use --overwrite to recompute)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, choices=["disentangled", "standard"],
                        help="AE variant used to generate encodings")
    parser.add_argument("--ae_ckpt",     required=True,
                        help="Path to trained Stage 1 .pth model")
    parser.add_argument("--poses",       required=True,
                        help="Path to dataset .pt file (train / dev / test)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory where per-sample .npy encodings will be saved")
    parser.add_argument("--max_frame_len", type=int, default=300,
                        help="Truncate sequences longer than this (default: 300 for PHOENIX, 350 for CSL)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Recompute encodings even if the .npy file already exists")
    main(parser.parse_args())
