"""
SLP Dataset — DARSLP / A²V-SLP
================================
PyTorch Dataset that serves (text embeddings, pose sequences, pose encodings)
tuples for Sign Language Production training.

Pose encodings (precomputed AE latents) are optional: pass encodings_path only
for Stage 2 training. Leave it None for Stage 1 (AE) training and inference.
"""

from torch.utils.data import Dataset
import numpy as np
import torch
import os


def pad_embeddings(a, max_len):
    """Truncate or zero-pad a 2D array along axis 0 to max_len."""
    if a.shape[0] >= max_len:
        return a[:max_len]
    pad = np.zeros((max_len - a.shape[0], a.shape[1]), dtype=a.dtype)
    return np.concatenate((a, pad), axis=0)


def pad_or_truncate_frames(a, max_len):
    """Truncate (randomly dropping frames) or pad (with value 2) to max_len."""
    n = a.shape[0]
    if n == max_len:
        return a
    if n > max_len:
        idx = np.sort(np.random.choice(n, max_len, replace=False))
        return a[idx]
    pad_shape = (max_len - n, *a.shape[1:])
    return np.concatenate((a, np.full(pad_shape, 2, dtype=a.dtype)), axis=0)


class SLPDataset(Dataset):
    """
    Dataset of BERT text embeddings, 3-D pose sequences, and (optionally)
    precomputed AE latent encodings for one dataset split.

    Args:
        pt_file_path:        Path to the .pt file with pose + metadata.
        text_embeddings_path: Directory containing per-sample BERT .npy files.
        texts_max_length:    Maximum token length for text padding.
        poses_max_length:    Maximum frame length for pose padding.
        encodings_path:      Directory with per-sample AE encoding .npy files.
                             Required for Stage 2 training; None otherwise.
        version_filter:      Optional list of version suffixes to restrict keys.
    """

    def __init__(self, pt_file_path, text_embeddings_path, texts_max_length,
                 poses_max_length, encodings_path=None, transform=None,
                 version_filter=None):
        self.data = torch.load(pt_file_path, weights_only=False)

        if version_filter is not None:
            self.keys = [k for k in self.data
                         if any(k.endswith(f"-{v}") for v in version_filter)]
        else:
            self.keys = list(self.data.keys())

        self.text_embeddings_path = text_embeddings_path
        self.texts_max_length     = texts_max_length
        self.poses_max_length     = poses_max_length
        self.encodings_path       = encodings_path
        self.transform            = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        key         = self.keys[idx]
        sample_data = self.data[key]

        # Text embeddings
        text_emb_file = os.path.join(self.text_embeddings_path, f"{key}.npy")
        if not os.path.exists(text_emb_file):
            raise FileNotFoundError(f"Text embedding not found: {text_emb_file}")
        text_embeddings = pad_embeddings(
            np.load(text_emb_file), self.texts_max_length
        ).astype(np.float32)

        # Pose sequence
        poses_3d    = sample_data["poses_3d"].numpy()
        frame_length = poses_3d.shape[0]
        poses_padded = pad_or_truncate_frames(poses_3d, self.poses_max_length).astype(np.float32)

        sample = {
            'name':             key,
            'text':             sample_data["text"],
            'gloss':            sample_data["gloss"],
            'speaker':          sample_data["speaker"],
            'text_embeddings':  text_embeddings,
            'poses':            poses_padded,
            'frame_lengths':    frame_length,
            'idx':              idx,
        }

        # Pose encodings (Stage 2 training only)
        if self.encodings_path is not None:
            enc_file = os.path.join(self.encodings_path, f"{key}.npy")
            if not os.path.exists(enc_file):
                raise FileNotFoundError(f"Pose encoding not found: {enc_file}")
            enc = np.load(enc_file)                          # (F, latent_dim)
            sample['pose_encodings'] = pad_embeddings(
                enc, self.poses_max_length
            ).astype(np.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample
